import math
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.nn.modules.batchnorm import BatchNorm1d

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Transpose1DBlock(nn.Module):
    """
    Class that represents a single convolutional block
    in the WaveGAN Generator network
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding=11,
        upsample=None, # Factor by which to upsample the feature map
        output_padding=1,
        use_batch_norm=False, # Batch normalization for feature map mini batch
        device=device
    ): 
        super().__init__()

        self.upsample = upsample
        self.device = device

        operations = []

        if self.upsample:
            operations.append(
                nn.Conv1d(
                    in_channels, out_channels,
                    kernel_size, stride,
                    padding='same',
                    device=self.device
                )
            )

        else:
            operations.append(
                nn.ConvTranspose1d(
                    in_channels, out_channels,
                    kernel_size, stride,
                    padding, output_padding,
                    device=self.device
                )
            )

        if use_batch_norm:
            operations.append(
                nn.BatchNorm1d(out_channels)
            )
        
        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        # As recommended in WaveGAN paper to use nearest neighbour upsampling
        if self.upsample:
            x = nn.functional.interpolate(
                x, scale_factor=self.upsample, mode='nearest')
            
        return self.operations(x)

class PhaseShuffle(nn.Module):
    """
    Performs phase shuffling, i.e. shifting feature axis of a 3D tensor
    by a random integer in {-n, n} and performing reflection padding where
    necessary.
    """
    def __init__(
        self,
        shift_factor,
        device=device):
        super().__init__()

        self.shift_factor = shift_factor
        self.device = device

    def forward(self, x):
        if self.shift_factor == 0:
            return x

        # Uniform distribution over axes in the range (-shift_factor, shift_factor)
        k_list = (
            torch.Tensor(
                x.shape[0]).random_(0, 2*self.shift_factor + 1) - self.shift_factor
        )
        k_list = k_list.numpy().astype(int)

        # Combine sample indices into lists so that less shuffle operations
        # need to be performed
        k_map = {}
        for idx, k in enumerate(k_list):
            k = int(k)
            if k not in k_map:
                k_map[k] = []
            
            k_map[k].append(idx)
        
        x_shuffle = x.clone()

        # Apply shuffle to each sample, applying reflective padding when necessary
        for k, idxs in k_map.items():
            if k > 0:
                x_shuffle[idxs] = F.pad(x[idxs][..., :-k], (k, 0), mode="reflect")
            else:
                x_shuffle[idxs] = F.pad(x[idxs][..., -k:], (0, -k), mode="reflect")
                
        assert x_shuffle.shape == x.shape


        return x_shuffle

class Conv1DBlock(nn.Module):
    """
    Convolution block used in the WaveGAN Discriminator network
    """
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        alpha=0.2,
        shift_factor=2,
        stride=4,
        padding=11, 
        use_batch_norm=True,
        dropout_prob=0,
        device=device
    ):
        super().__init__()

        self.alpha = alpha
        self.device = device

        # Convolutional layer
        operations = [
            nn.Conv1d(
            in_channels, out_channels,
            kernel_size, 
            stride=stride, 
            padding=padding,
            device=self.device)]

        # Batch normalization layer
        if use_batch_norm:
            operations.append(
                nn.BatchNorm1d(out_channels)
            )
        
        # Activation function
        operations.append(
            nn.LeakyReLU()
        )

        # Phase shuffling
        operations.append(
            PhaseShuffle(
                shift_factor,
                device=self.device)
        )

        # Dropout for training phase
        if dropout_prob > 0:
            operations.append(
                nn.Dropout(dropout_prob)
            )

        self.operations = nn.Sequential(*operations)

    def forward(self, x):
        return self.operations(x)

class WaveGANGenerator(nn.Module):
    """
    Generator network for WaveGAN
    Adapted to use explicit latent variable and output dimensions
    """
    def __init__(
        self,
        in_dim,
        out_dim,
        kernel_size=25,
        # num_channels=[1024, 512, 256, 128, 64], # Number of channels of filters for each convolutional block
        num_channels=[1, 512, 256, 128, 64],
        stride=4,
        upsample=2, # Default 2
        use_batch_norm=False,
        device=device):
        super().__init__()

        # Ensure power of 2 for computation of dimensions
        assert math.ceil(np.log2(out_dim)) == math.floor(np.log2(out_dim))

        self.in_dim = in_dim
        self.out_dim = out_dim

        self.use_batch_norm = use_batch_norm
        self.upsample = upsample
        self.stride = 1 if upsample else stride
        self.device = device

        # FC layer
        self.fc_out_channels = num_channels[0]
        self.fc_length = (self.out_dim // (self.upsample ** len(num_channels)))
        fc_out_dim = self.fc_out_channels * self.fc_length
        fc_block = [
            nn.Linear(self.in_dim, fc_out_dim)
        ]

        if self.use_batch_norm:
            fc_block.append(
                nn.BatchNorm1d(self.fc_out_channels)
            )

        fc_block.append(
            nn.ReLU()
        )

        self.fc_block = nn.Sequential(*fc_block)

        # Convolutional layers
        conv_blocks = []
        for in_channels, out_channels in zip(num_channels, num_channels[1:]):
            conv_blocks.append(
                Transpose1DBlock(
                    in_channels, out_channels,
                    kernel_size=kernel_size,
                    stride=self.stride,
                    upsample=self.upsample,
                    use_batch_norm=self.use_batch_norm,
                    device=self.device
                )
            )
            conv_blocks.append(
                nn.ReLU()
            )
        # Final convolutional block to reduce to single channel waveform
        conv_blocks.append(
            Transpose1DBlock(
                num_channels[-1], 1,
                kernel_size=kernel_size,
                stride=self.stride,
                upsample=self.upsample,
                use_batch_norm=self.use_batch_norm,
                device=self.device
            )
        )
        conv_blocks.append(
            # Normalize between -1 and 1 to generate amplitude normalized signal waveforms
            nn.Tanh()
        )
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Normalize initialization parameters
        for m in self.modules():
            if isinstance(m, nn.ConvTranspose1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.fc_block(x)
        x = x.view(-1, self.fc_out_channels, self.fc_length)
        return self.conv_blocks(x)


class WaveGANDiscriminator(nn.Module):
    """
    Discriminator network for WaveGAN
    Adapted to use explicit waveform signal dimensions based on the output from the generator network
    """
    def __init__(
        self,
        in_dim,
        num_channels = [1, 64, 128, 256, 512, 1024],
        kernel_size=25,
        shift_factor=2,
        alpha=0.2,
        stride=4,
        padding=11,
        use_batch_norm=True,
        device=device) -> None:
        super().__init__()

        self.in_dim = in_dim
        self.stride = stride
        self.padding = padding
        self.shift_factor = shift_factor
        self.use_batch_norm = use_batch_norm
        self.device = device
        self.alpha = alpha
        self.kernel_size = kernel_size

        # Convolutional blocks
        conv_blocks = []
        for in_channels, out_channels in zip(num_channels, num_channels[1:]):
            conv_blocks.append(
                Conv1DBlock(
                    in_channels, out_channels,
                    kernel_size=self.kernel_size,
                    alpha=self.alpha,
                    shift_factor=self.shift_factor,
                    stride=self.stride if out_channels != num_channels[-1] else self.stride//2, # Halve the stride for last conv block
                    padding=self.padding if out_channels != num_channels[-1] else self.padding + 1,
                    device=self.device
                )
            )
        conv_blocks.append(nn.Flatten())    
        self.conv_blocks = nn.Sequential(*conv_blocks)

        # Linear layer with sigmoid to calculate probabilities 
        self.fc_in_dim = self.in_dim // (self.stride ** (len(num_channels) - 2))
        self.fc_in_dim = self.fc_in_dim // (self.stride // 2)
        fc_block = [
            nn.Linear(self.fc_in_dim * num_channels[-1], 1, device=self.device),
            nn.Sigmoid()
        ]
        self.fc_block = nn.Sequential(*fc_block)

        # Normalize initialization parameters
        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight.data)

    def forward(self, x):
        x = self.conv_blocks(x)
        return self.fc_block(x)
