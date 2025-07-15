
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from .utils import get_padded_size, spatial_pad_crop
from .nn import Conv, ResBlockWithResampling, BottomUpLayer, TopDownLayer


import torch
import torch.nn as nn
from torch.nn import functional as F
import torch
import torch.nn as nn
import torch.nn.functional as F


class UNet(nn.Module):
    """
    UNet - Denoising Convolutional Neural Network
    
    Architecture based on the paper:
    "Beyond a Gaussian Denoiser: Residual Learning of Deep CNN for Image Denoising"
    
    Args:
        channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        num_layers (int): Number of layers in the network
        features (int): Number of feature maps
        kernel_size (int): Size of convolutional kernels
    """
    def __init__(self, channels=3, num_layers=17, features=64, kernel_size=3):
        super(UNet, self).__init__()
        padding = kernel_size // 2
        layers = []
        
        # First layer: Conv + ReLU
        layers.append(nn.Conv2d(in_channels=channels, out_channels=features,
                                kernel_size=kernel_size, padding=padding, bias=False))
        layers.append(nn.ReLU(inplace=True))
        
        # Middle layers: Conv + BN + ReLU
        for _ in range(num_layers - 2):
            layers.append(nn.Conv2d(in_channels=features, out_channels=features,
                                    kernel_size=kernel_size, padding=padding, bias=False))
            layers.append(nn.BatchNorm2d(features))
            layers.append(nn.ReLU(inplace=True))
            
        # Last layer: Conv (without activation)
        layers.append(nn.Conv2d(in_channels=features, out_channels=channels,
                                kernel_size=kernel_size, padding=padding, bias=False))
        
        self.UNet = nn.Sequential(*layers)
        
        # Initialize weights
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        # UNet predicts the noise rather than the denoised image
        noise = self.UNet(x)
        # Subtract the predicted noise from the input to get the denoised image
        out = x - noise
        return out

class ADNet(nn.Module):
    """
    ADNet - Attention-guided Denoising Network
    
    Architecture incorporating spatial and channel attention mechanisms for improved denoising.
    
    Args:
        in_channels (int): Number of input channels (1 for grayscale, 3 for RGB)
        out_channels (int): Number of output channels
        num_features (int): Number of feature maps
    """
    def __init__(self, in_channels=3, out_channels=3, num_features=64):
        super(ADNet, self).__init__()
        
        # Initial feature extraction
        self.conv_first = nn.Conv2d(in_channels, num_features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=True)
        
        # Main residual blocks with attention
        self.res_blocks = nn.ModuleList([
            ResidualAttentionBlock(num_features) for _ in range(8)
        ])
        
        # Global residual learning path
        self.global_skip_conv = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        
        # Final reconstruction
        self.final_conv = nn.Conv2d(num_features, out_channels, kernel_size=3, padding=1)
        
    def forward(self, x):
        # Initial feature extraction
        feat = self.relu(self.conv_first(x))
        feat_orig = feat
        
        # Process through residual attention blocks
        for block in self.res_blocks:
            feat = block(feat)
        
        # Global residual connection
        feat = self.global_skip_conv(feat) + feat_orig
        
        # Final reconstruction (predict the clean image directly)
        out = self.final_conv(feat)
        
        return out


class ChannelAttention(nn.Module):
    """Channel attention module"""
    def __init__(self, num_features, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        
        self.fc = nn.Sequential(
            nn.Conv2d(num_features, num_features // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(num_features // reduction, num_features, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial attention module"""
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        # Generate attention map using max and average pooling along channel dimension
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        attention = torch.cat([avg_out, max_out], dim=1)
        attention = self.sigmoid(self.conv(attention))
        
        return attention * x


class ResidualAttentionBlock(nn.Module):
    """Residual block with both channel and spatial attention"""
    def __init__(self, num_features):
        super(ResidualAttentionBlock, self).__init__()
        
        self.conv1 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_features)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv2d(num_features, num_features, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_features)
        
        # Attention modules
        self.ca = ChannelAttention(num_features)
        self.sa = SpatialAttention()
        
    def forward(self, x):
        residual = x
        
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        
        # Apply attention
        out = self.ca(out)
        out = self.sa(out)
        
        # Residual connection
        out += residual
        out = self.relu(out)
        
        return out

'''
class UNet(nn.Module):
    """UNet model with self-attention.
    
    Args:
        colour_channels (int): Number of colour channels in the input image.
        blocks_per_layer (int): Number of residual blocks per layer.
        n_filters (int): Number of filters in the convolutional layers.
        n_layers (int): Number of layers in the UNet.
        td_skip (bool): Whether to use skip connections in the top-down pass.
        downsampling (list): Number of downsampling steps per layer.
        loss_fn (str): Loss function to use. Default: 'MSE'.
        checkpointed (bool): Whether to use activation checkpointing in the forward pass.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
        attention_layers (list): Indices of layers to add attention to.
    """

    def __init__(
        self,
        colour_channels=3,
        blocks_per_layer=1,
        n_filters=64,
        n_layers=14,
        td_skip=True,
        downsampling=None,
        loss_fn="MSE",
        checkpointed=False,
        dimensions=2,
        attention_layers=None,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.blocks_per_layer = blocks_per_layer
        self.n_filters = n_filters
        self.td_skip = td_skip
        if loss_fn not in ("MSE", "L1"):
            raise ValueError("UNet loss_fn should be either 'L1' or 'MSE'")
        self.loss_fn = loss_fn
        self.checkpointed = checkpointed
        
        # Set default attention_layers if None
        if attention_layers is None:
            # Add attention to middle and deeper layers (25% of the total)
            attention_layers = list(range(n_layers - n_layers // 32, n_layers))
        self.attention_layers = attention_layers

        # Number of downsampling steps per layer
        if downsampling is None:
            downsampling = [0] * self.n_layers
        self.n_downsample = sum(downsampling)

        assert max(downsampling) <= self.blocks_per_layer
        assert len(downsampling) == self.n_layers

        self.first_bottom_up = nn.Sequential(
            Conv(
                colour_channels,
                n_filters,
                5,
                padding=2,
                padding_mode="replicate",
                dimensions=dimensions,
            ),
            nn.ReLU(inplace=True),
            ResBlockWithResampling(
                c_in=n_filters,
                c_out=n_filters,
                gated=False,
                dimensions=dimensions,
            ),
        )

        self.top_down_layers = nn.ModuleList()
        self.bottom_up_layers = nn.ModuleList()

        for i in range(self.n_layers):
            is_top = i == self.n_layers - 1
            use_attention = i in self.attention_layers

            self.bottom_up_layers.append(
                BottomUpLayer(
                    n_res_blocks=self.blocks_per_layer,
                    n_filters=n_filters,
                    downsampling_steps=downsampling[i],
                    dimensions=dimensions,
                    attention=use_attention
                )
            )

            self.top_down_layers.append(
                TopDownLayer(
                    n_res_blocks=blocks_per_layer,
                    n_filters=n_filters,
                    is_top_layer=is_top,
                    upsampling_steps=downsampling[i],
                    skip=td_skip,
                    dimensions=dimensions,
                    attention=use_attention
                )
            )

        self.final_top_down = Conv(
            in_channels=n_filters,
            out_channels=colour_channels,
            kernel_size=1,
            dimensions=dimensions,
        )

    def forward(self, x):
        # Pad x to have base 2 side lengths to make resampling steps simpler
        # Save size to crop back down later
        x_size = x.size()[2:]
        padded_size = get_padded_size(x_size, self.n_downsample)
        x = spatial_pad_crop(x, padded_size)

        bu_values = self.bottomup_pass(x)

        output = self.topdown_pass(bu_values)

        # Restore original image size
        output = spatial_pad_crop(output, x_size)
        return output

    def bottomup_pass(self, x):
        # Bottom-up initial layer
        x = self.first_bottom_up(x)

        # Loop from bottom to top layer, store all deterministic nodes we
        # need in the top-down pass
        bu_values = []
        for i in range(self.n_layers):
            if i % 2 == 0 and self.checkpointed:
                x = checkpoint(
                    self.bottom_up_layers[i],
                    x,
                    use_reentrant=False,
                )
            else:
                x = self.bottom_up_layers[i](x)
            bu_values.append(x)

        return bu_values

    def topdown_pass(
        self,
        bu_values,
    ):
        out = None
        for i in reversed(range(self.n_layers)):
            skip_input = out

            if i % 2 == 0 and self.checkpointed:
                out = checkpoint(
                    self.top_down_layers[i],
                    out,
                    skip_input,
                    bu_values[i],
                    use_reentrant=False,
                )
            else:
                out = self.top_down_layers[i](
                    out,
                    skip_input,
                    bu_values[i],
                )

        out = self.final_top_down(out)
        out = torch.sigmoid(out)
        return out

    def loss(self, x, y):
        if self.loss_fn == "L1":
            return F.l1_loss(x, y, reduction="none")
        elif self.loss_fn == "MSE":
            return F.mse_loss(x, y, reduction="none")
        
        
'''