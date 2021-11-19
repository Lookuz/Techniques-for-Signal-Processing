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
    
    x --> Conv3 --> WaveNetCycle1 --> WaveNetCycle2 -- + --> Linear(ReLU) --> Linear(ReLU)
                                            |          |
                                            | ________ |
    """
    def __init__(
        self,
        in_channels,
        out_dim,
        wavenet_channels=128,
        wavenet_kernel_size=9,
        dilations=[2**i for i in range(10)],
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

    def forward(self, x):
        # First Conv3
        out = self.pre_wavenet_conv(x)

        # WaveNet Cycle computation
        # Use the output of the first WaveNet Cycle as a residual connection
        # With the second WaveNetCycle
        x_in = out
        out_skip_1 = torch.zeros_like(x_in)
        for wavenet in self.wavenet_cycle_1:
            x_in, x_skip = wavenet(x_in)
            out_skip_1 += x_skip

        # Second WaveNet cycle
        # Use direct output of the previous WaveNet cycle as the input
        out_skip_2 = torch.zeros_like(x_in)
        for wavenet in self.wavenet_cycle_2:
            x_in, x_skip = wavenet(x_in)
            out_skip_2 += x_skip

        wavenet_out = out_skip_1 + out_skip_2

        return wavenet_out
