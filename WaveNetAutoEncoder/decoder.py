from typing import ForwardRef
import numpy as np
import torch
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class CausalConv1d(nn.Conv1d):
    """
    Causal 1D Convolutional layer that augments the existing
    Conv1d module in PyTorch to use Causal Convolutions instead
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        bias=True,
        device=device) -> None:
        super(CausalConv1d, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            padding=((kernel_size - 1) * dilation), # Extra padding to use causal convolutions
            stride=stride,
            dilation=dilation,
            bias=bias,
            device=device
        )

    def forward(self, x):
        out = super(CausalConv1d, self).forward(x)

        # Padding applies to left and right -> Remove redundant right padding
        return out[:, :, :-self.padding[0]] 

class WaveNetLayer(nn.Module):
    """
    Module that represents a single WaveNet layer that includes
    Gated Causal Convolutional Layer with residual connections.
    A visual representation is given below:

         | -----------------------Residual---------------------------|
         |                                                           |
         |      | -- CausalConv1d -- TanH ----- |                    |
    x -- | ---- |                               * ---- | -- 1 * 1 -- + -- output
         |      | -- CausalConv1d -- Sigmoid -- |      |
                                                     1 * 1
                                                       |
                                                       |
      -------- Accumulated skip connected inputs ----- + ----------------->                                                 +
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        dilation=1,
        device=device) -> None:
        super().__init__()

        # Convolution filter component
        self.conv_filter = CausalConv1d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            device=device
        )
        self.activation_filter = nn.Tanh()

        # Gated convolution component
        self.conv_gate = CausalConv1d(
            in_channels, out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            device=device
        )
        self.activation_gate = nn.Sigmoid()

        # Residual connections
        self.residual_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )

        # Skip connections
        self.skip_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, bias=False
        )

    def forward(self, x):
        # Gated convolution computation
        out_filter = self.conv_filter(x)
        out_filter = self.activation_filter(out_filter)
        out_gate = self.conv_gate(x)
        out_gate = self.activation_gate(out_gate)
        out_hidden = out_gate * out_filter

        # Residual connection
        out_residual = self.residual_conv(out_hidden) + x

        # Skip connection
        out_skip = self.skip_conv(out_hidden)

        return out_residual, out_skip

class WaveNetDecoder(nn.Module):
    """
    Module that represents a decoder using the WaveNet cycles
    Visualisation of decoder pipeline
    
    x --> Conv3 --> WaveNetCycle1 --> WaveNetCycle2 --  + --> Linear(ReLU) --> Linear(ReLU)
                                            |           |
                                            | _ _ _ _ _ |
    """
    def __init__(
        self,
        in_channels,
        latent_dim,
        out_dim,
        wavenet_channels=128,
        wavenet_kernel_size=9,
        dilations=[2**i for i in range(10)],
        # Parameters for deconvolution block
        init_kernel_size=4,
        deconv_stride=2,
        device=device) -> None:
        super().__init__()

        self.device = device

        # Conv3
        self.pre_wavenet_conv = nn.Conv1d(
            in_channels, wavenet_channels,
            kernel_size=3, padding=1, device=self.device
        )
        
        # WaveNet Cycles
        self.wavenet_cycle_1 = nn.ModuleList()
        self.wavenet_cycle_2 = nn.ModuleList()
        for d in dilations:
            self.wavenet_cycle_1.append(
                WaveNetLayer(
                    wavenet_channels, wavenet_channels, 
                    wavenet_kernel_size, 
                    dilation=d,
                    device=self.device
                )
            )
            self.wavenet_cycle_2.append(
                WaveNetLayer(
                    wavenet_channels, wavenet_channels, 
                    wavenet_kernel_size, 
                    dilation=d,
                    device=self.device
                )
            )

        # Upsampling and deconvolution
        num_channels = wavenet_channels
        kernel_size = init_kernel_size
        conv_transpose_blocks = []
        num_expansions = int(np.log2(out_dim / latent_dim))
        for i in range(num_expansions):
            conv_transpose_blocks.append(
                nn.ConvTranspose1d(
                    num_channels, num_channels//2 if i < num_expansions - 1 else 1, 
                    kernel_size=kernel_size, 
                    stride=deconv_stride,
                    padding = (kernel_size - deconv_stride)//2
                )
            )
            kernel_size *= 2
            num_channels //= 2

        self.conv_transpose_blocks = nn.Sequential(*conv_transpose_blocks)

    def forward(self, x):
        # First Conv3
        out = self.pre_wavenet_conv(x)

        # WaveNet Cycle computation
        # Use the output of the first WaveNet Cycle as a residual connection
        # With the second WaveNetCycle
        x_in = out
        out_skip_1 = torch.zeros_like(x_in)
        for wavenet in self.wavenet_cycle_1:
            x_in = wavenet(x_in)
            # out_skip_1 += x_skip
        
        out_skip_1 = x_in

        # Second WaveNet cycle
        # Use direct output of the previous WaveNet cycle as the input
        out_skip_2 = torch.zeros_like(x_in)
        for wavenet in self.wavenet_cycle_2:
            x_in = wavenet(x_in)
            # out_skip_2 += x_skip
        
        out_skip_2 = x_in

        wavenet_out = out_skip_1 + out_skip_2

        # Deconvolution
        out = self.conv_transpose_blocks(wavenet_out)

        return wavenet_out
