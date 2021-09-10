from typing import final
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
        C_in, # Number of channels for input
        C_out, # Number of channels for output
        # Default parameters for the SK module
        kernels=[9, 16], # 2 feature maps with kernel sizes 9 and 16
        fc_shared_sizes=[8], # Shared MLP hidden layer sizes
        fc_indep_sizes=[], # Independent output layer sizes
        activation_fn=nn.ReLU
        ):
        super().__init__()

        self.C_in = C_in
        self.C_out = C_out

        self.kernels = kernels
        self.fc_shared_sizes = [self.C_out] + fc_shared_sizes # Include input size
        self.fc_indep_sizes = [self.fc_shared_sizes[-1]] + fc_indep_sizes + [self.C_out]
        self.activation_fn = activation_fn()
        self.softmax = nn.Softmax(dim=0)

        # Feature maps
        self.conv_layers = nn.ModuleList()
        for k in self.kernels:
            self.conv_layers.append(
                nn.Conv1d(
                    C_in, C_out, # In/Out channels
                    kernel_size=k, 
                    padding='same' # Same size output
                )
            )

        # MLP
        self.mlp = nn.ModuleList()
        for in_size, out_size in zip(self.fc_shared_sizes, self.fc_shared_sizes[1:]):
            self.mlp.append(nn.Linear(in_size, out_size))

        # Independent output heads
        self.output_heads = nn.ModuleList()
        for _ in range(len(self.kernels)):
            # Create each output network
            output_head = []
            for in_size, out_size in zip(self.fc_indep_sizes, self.fc_indep_sizes[1:]):
                output_head.append(nn.Linear(in_size, out_size)) 

            self.output_heads.append(
                nn.Sequential(*output_head)
            )

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

# Singular block in the SKCNN network architecture
class SKCNNBlock(nn.Module):
    def __init__(
        self,
        W,
        C_in,
        C_out,
        # SK parameters
        kernels=[9, 16],
        fc_shared_sizes=[8], 
        fc_indep_sizes=[], 
        activation_fn=nn.ReLU,
        # Pooling parameters
        pooling_size=7,
        stride=7
        ):
        super().__init__()

        # Selective Kernel
        self.selective_kernel = SelectiveKernel(
            W, C_in, C_out,
            kernels=kernels,
            fc_shared_sizes=fc_shared_sizes,
            fc_indep_sizes=fc_indep_sizes,
            activation_fn=activation_fn
        )

        # Max pooling
        self.max_pooling = nn.MaxPool1d(
            kernel_size=pooling_size,
            stride=stride
        )

        # Batch normalization
        self.batch_norm = nn.BatchNorm1d(C_in)

    def forward(self, x):
        # Selective kernel 
        output = self.selective_kernel(x)
        output = output.unsqueeze(0)

        # Max pooling
        output = self.max_pooling(output)

        # Batch normalization
        output = self.batch_norm(output)

        return output

# SKCNN Network
class SKCNN(nn.Module):
    def __init__(self,
        W,
        C_in,
        num_classes=11,
        # SK block parameters - In order of blocks
        sk_block_params = [
            {
                'C_out' : 16,
                'kernels' : [9, 16],
                'fc_shared_sizes' : [16],
                'fc_indep_sizes' : [],
                'activation_fn' : nn.ReLU,
                'pooling_size' : 7,
                'stride' : 7
            },
            {
                'C_out' : 32,
                'kernels' : [9, 16],
                'fc_shared_sizes' : [32],
                'fc_indep_sizes' : [],
                'activation_fn' : nn.ReLU,
                'pooling_size' : 7,
                'stride' : 7
            },
            {
                'C_out' : 64,
                'kernels' : [9, 16],
                'fc_shared_sizes' : [64],
                'fc_indep_sizes' : [],
                'activation_fn' : nn.ReLU,
                'pooling_size' : 7,
                'stride' : 7
            },
            {
                'C_out' : 128,
                'kernels' : [9, 16],
                'fc_shared_sizes' : [128],
                'fc_indep_sizes' : [],
                'activation_fn' : nn.ReLU,
                'pooling_size' : 7,
                'stride' : 7
            }
        ],
        # FC Layer
        fc_block_sizes=[512]):
        super().__init__()

        # SKCNN Blocks
        skcnn_blocks = []
        input_channels = C_in
        for params in sk_block_params:
            skcnn_blocks.append(
                SKCNNBlock(
                    W,
                    input_channels,
                    params['C_out'],
                    fc_shared_sizes=params['fc_shared_sizes'],
                    fc_indep_sizes=params['fc_indep_sizes'],
                    activation_fn=params['activation_fn'],
                    pooling_size=params['pooling_size'],
                    stride=params['stride']
                )
            )
            input_channels = params['C_out']
        
        self.skcnn_blocks = nn.Sequential(*skcnn_blocks)

        # FC Block
        fc_block_input_size = self.compute_fc_block_input_size(W, sk_block_params=sk_block_params)
        fc_sizes = [fc_block_input_size] + fc_block_sizes

        fc_layers = []
        for i, (in_size, out_size) in enumerate(zip(fc_sizes, fc_sizes[1:])):
            fc_layers.append(nn.Linear(
                in_size, out_size
            )) 

            # Only add ReLU for more than one hidden layer case
            if i > 0:
                fc_layers.append(nn.ReLU())

        fc_layers.append(nn.Linear(
            fc_sizes[-1], num_classes
        ))
        fc_layers.append(nn.Softmax())

        self.fc = nn.Sequential(*fc_layers)


    # Formula take from: https://pytorch.org/docs/stable/generated/torch.nn.MaxPool1d.html
    def compute_fc_block_input_size(
        self, W, sk_block_params):

        final_size = W
        out_channels = 0
        for params in sk_block_params:
            final_size = int((final_size - params['pooling_size'])/params['stride'] + 1)
            out_channels = params['C_out']
        
        final_size *= out_channels

        return final_size


    def forward(self, x):
        # SKCNN Blocks
        output = self.skcnn_blocks(x)
        output = torch.flatten(output, 1)

        # FC Block
        output = self.fc(output)

        return output
