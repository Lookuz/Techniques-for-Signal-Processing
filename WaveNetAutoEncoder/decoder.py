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
      -------- Accumulated skip connected inputs ----- + ----------------->

    NOTE: Due to the nature of the implementation, may cause memory bloat from stray skip connection variables
          Recommended to use the WaveNetCycle class implemented below that encapsulates and localizes logic for skip connections
          within the forward function
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

        self.device = device

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
            kernel_size=1, bias=False,
            device=self.device
        )

        # Skip connections
        self.skip_conv = nn.Conv1d(
            in_channels, out_channels,
            kernel_size=1, bias=False,
            device=self.device
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

class WaveNetCycle(nn.Module):
    """
    Module that represents a WaveNetCycle consisting of sequential application
    of WaveNet units using Gated Causal Convolutional Layer with residual connections.
    The number of WaveNet units is dependent on the number of dilations used
    
    A visual representation of a WaveNet unit is given below(Repeated from WaveNetLayer):

         | -----------------------Residual---------------------------|
         |                                                           |
         |      | -- CausalConv1d -- TanH ----- |                    |
    x -- | ---- |                               * ---- | -- 1 * 1 -- + -- output
         |      | -- CausalConv1d -- Sigmoid -- |      |
                                                     1 * 1
                                                       |
                                                       |
      -------- Accumulated skip connected inputs ----- + ----------------->

    """
    def __init__(
        self,
        wavenet_channels,
        kernel_size,
        dilations=[2**i for i in range(10)],
        stride=1,
        device=device):
        super().__init__()

        self.device = device
        self.dilations = dilations

        # Filter convolutional layer
        self.conv_filter = nn.ModuleList()
        self.activation_filter = nn.Tanh()
        
        # Gated convolutional layer
        self.conv_gate = nn.ModuleList()
        self.activation_gate = nn.Sigmoid()
        
        # Residual convolution layer
        self.conv_residual = nn.ModuleList()

        # Skip connection convolution
        self.conv_skip = nn.ModuleList()

        # Populate filter and gated convolutional layers with CausalConv1d
        # Skip and residual connections uses 1x1 convolutional operations
        for d in self.dilations:
            self.conv_filter.append(
                CausalConv1d(
                    wavenet_channels, wavenet_channels,
                    kernel_size, 
                    stride=stride,
                    dilation=d,
                    device=self.device
                )
            )
            self.conv_gate.append(
                CausalConv1d(
                    wavenet_channels, wavenet_channels,
                    kernel_size, 
                    stride=stride,
                    dilation=d,
                    device=self.device
                )
            )
            self.conv_residual.append(
                nn.Conv1d(
                    wavenet_channels, wavenet_channels,
                    kernel_size=1, bias=False,
                    device=self.device
                )
            )
            self.conv_skip.append(
                nn.Conv1d(
                    wavenet_channels, wavenet_channels,
                    kernel_size=1, bias=False,
                    device=self.device
                )
            )

    def forward(self, x):

        x_in = x
        out_skip = torch.zeros_like(x)

        for i in range(len(self.dilations)):
            # Filter and gated convolutions
            f = self.conv_filter[i](x_in)
            f = self.activation_filter(f)
            g = self.conv_gate[i](x_in)
            g = self.activation_gate(g)
            x_in = f * g

            # Accumulate skip connections
            out_skip += self.conv_skip[i](x_in)

            # Residual connection between input and gated convolution
            x_in = self.conv_residual[i](x_in) + x_in
        
        return x_in, out_skip

class WaveNetDecoder(nn.Module):
    """
    Module that represents a decoder using the WaveNet cycles
    Visualisation of decoder pipeline
    
    x --> Conv3 --> WaveNetCycle1 --> WaveNetCycle2 --  + --> ConvTranpose1d Block
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
        self.wavenet_cycle_1 = WaveNetCycle(
            wavenet_channels, 
            wavenet_kernel_size,
            dilations=dilations,
            device=self.device
        )
        self.wavenet_cycle_2 = WaveNetCycle(
            wavenet_channels, 
            wavenet_kernel_size,
            dilations=dilations,
            device=self.device
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
                    padding = (kernel_size - deconv_stride)//2,
                    device=self.device
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
        x_in, out_skip_1 = self.wavenet_cycle_1(out)

        # Second WaveNet cycle
        x_in, out_skip_2 = self.wavenet_cycle_1(x_in)

        wavenet_out = out_skip_1 + out_skip_2

        # Deconvolution
        out = self.conv_transpose_blocks(wavenet_out)

        return out
