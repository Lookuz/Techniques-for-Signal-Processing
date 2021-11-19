import numpy as np
import torch
from torch import nn

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

class ConvReLUBlock(nn.Module):
    """
    Module representing a single encoder convolutional block
    in the WaveNetEncoder
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        residual=True,
        device=device):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.residual = residual
        if self.residual:
            assert self.stride == 1, "Residual connections between convolutional maps only available for stride 1"
            assert self.kernel_size % 2 != 0, "Residual connections between convolutional maps only available for odd kernel sizes"
            assert self.in_channels == self.out_channels, "Residual connections between convolutional maps only available for same input and output channels"
        self.device = device

        self.conv = nn.Conv1d(
            self.in_channels, self.out_channels,
            self.kernel_size,
            self.stride,
            padding=0,
            device=self.device
        )

        self.relu = nn.ReLU()
    
    def forward(self, x):
        out = self.conv(x)
        out = self.relu(out)

        # Compute kernel offsets
        offset = (self.kernel_size - 1) // 2

        # Residual connection
        if self.residual:
            out += x[:, :, offset:-offset]

        return out

class VAEBottleNeck(nn.Module):
    """offset
    Module representing the Variational Autoencoder(VAE) bottleneck segment
    """
    def __init__(
        self,
        in_channels,
        in_dim,
        out_dim,
        device=device):
        super().__init__()

        self.device = device
        self.in_channels = in_channels
        self.in_dim = in_dim
        self.out_dim = out_dim
        
        # Feature vector mean
        self.conv_mu = nn.Conv1d(
            self.in_channels, 1,
            kernel_size=1, device=self.device
        )
        self.linear_mu = nn.Linear(
            self.in_dim, self.out_dim, device=self.device
        )
        # Feature vector variance
        self.conv_sigma = nn.Conv1d(
            self.in_channels, 1,
            kernel_size=1, device=self.device
        )
        self.linear_sigma = nn.Linear(
            self.in_dim, self.out_dim, device=self.device
        )
        self.activation = nn.Tanh()

    def forward(self, x):
        # Latent representation vector
        mu = self.conv_mu(x)
        mu = self.linear_mu(mu)
        mu = self.activation(mu)

        # Latent representation variance
        sigma = self.conv_sigma(x)
        sigma = self.linear_sigma(sigma)
        sigma = torch.exp(0.5 * sigma) # Ensure PSD

        # Training step - Add randomness to feature vector
        latent = mu
        if self.train:
            epsilon = torch.randn(x.shape[0], 1, 1, device=self.device)
            latent += sigma * epsilon

        return latent

class WaveNetEncoder(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        # TODO: Consider stagger the increasing number of channels for stride 2
        block_config=[
            {'in_channels':1, 'out_channels':256, 'kernel_size':9, 'stride':1, 'residual':False},
            {'in_channels':256, 'out_channels':256, 'kernel_size':9, 'stride':1, 'residual':True},
            {'in_channels':256, 'out_channels':256, 'kernel_size':15, 'stride':2, 'residual':False},
            {'in_channels':256, 'out_channels':256, 'kernel_size':9, 'stride':1, 'residual':True},
            {'in_channels':256, 'out_channels':256, 'kernel_size':9, 'stride':1, 'residual':True},
            {'in_channels':256, 'out_channels':256, 'kernel_size':15, 'stride':2, 'residual':False},
            {'in_channels':256, 'out_channels':256, 'kernel_size':9, 'stride':1, 'residual':True},
            {'in_channels':256, 'out_channels':256, 'kernel_size':9, 'stride':1, 'residual':True},
            # {'in_channels':256, 'out_channels':256, 'kernel_size':9, 'stride':1, 'residual':True}
        ],
        device=device):
        super().__init__()

        self.device = device
        self.in_dim = in_dim
        assert self.in_dim % 2 == 0, "Signal length should be multiple of 2 to facilitate matching of reconstructed signal dimensions"
        self.out_dim = out_dim
        self.block_config = block_config

        encoder_blocks = []
        conv_feat_size = self.in_dim
        for config in block_config:
            encoder_blocks.append(
                ConvReLUBlock(
                    config['in_channels'], config['out_channels'],
                    config['kernel_size'], config['stride'],
                    config['residual'], self.device
                )
            )
            # Compute feature map length
            # Taken from https://pytorch.org/docs/stable/generated/torch.nn.Conv1d.html
            conv_feat_size = ((conv_feat_size - config['kernel_size']) // config['stride']) + 1

        self.encoder_blocks = nn.Sequential(*encoder_blocks)

        # VAE Module
        self.vae = VAEBottleNeck(
            block_config[-1]['out_channels'],
            conv_feat_size,
            self.out_dim,
            device=self.device
        )


    def forward(self, x):
        # TODO: Consider using the extracted feature maps from encoder blocks as inputs into decoder
        #       and feature vector separately for separate multi task training?
        feat = self.encoder_blocks(x)
        return self.vae(feat)
