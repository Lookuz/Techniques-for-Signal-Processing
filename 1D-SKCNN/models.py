import numpy as np
import torch 
from torch import nn
from torch.nn.modules.container import ModuleList

#TODO: Add capability for GPUs

# Selective Kernel module as described in the paper.
# Consists of multiple feature maps produced from several 1D convolutional layers,
# which are combined using soft attention using weights learned from feature maps
class SelectiveKernel(nn.Module):
    def __init__(
        self,
        W, # 1D vector size(Max width of an intrapulse signal)
        C, # Number of channels of input
        # Default parameters for the SK module
        kernels=[9, 16], # 2 feature maps with kernel sizes 9 and 16
        fc_layer_sizes=[512], # 1 FC layer with 512 neurons
        activation_fn=nn.ReLU
        ):
        super().__init__()

        self.kernels = kernels
        self.fc_layer_sizes = [C] + fc_layer_sizes # Include input size
        self.activation_fn = activation_fn()
        self.softmax = nn.Softmax(dim=0)

        # Feature maps
        self.conv_layers = nn.ModuleList()
        for k in self.kernels:
            self.conv_layers.append(
                nn.Conv1d(
                    C, C, # In/Out channels
                    kernel_size=k, 
                    padding='same' # Same size output
                )
            )

        # MLP
        self.mlp = nn.ModuleList()
        for in_size, out_size in zip(self.fc_layer_sizes, self.fc_layer_sizes[1:]):
            self.mlp.append(nn.Linear(in_size, out_size))

        # Output heads
        self.output_heads = nn.ModuleList()
        for _ in range(len(self.kernels)):
            self.output_heads.append(nn.Linear(fc_layer_sizes[-1], C)) 

    def forward(self, x):
        # Feature maps via 1D convolution
        feature_maps = [
            conv_layer(x) for conv_layer in self.conv_layers
        ]

        # Additive fusion 
        feature_maps_add = sum(feature_maps)

        # Global average pooling
        output = torch.mean(feature_maps_add, dim=-1)

        # Feature map weighting
        for fc in self.mlp:
            output = self.activation_fn(fc(output))

        weights = torch.cat([
            fc(output) for fc in self.output_heads
        ])
        weights = self.softmax(weights)

        # Soft attention via convex combination with weights
        feature_maps = torch.cat(feature_maps)
        output = weights[:, :, None] * feature_maps
        output = torch.sum(output, dim=0)
        
        return output
