import torch
from torch import nn
from torch.utils.checkpoint import checkpoint
from torch.nn import functional as F
from .utils import get_padded_size, spatial_pad_crop


class SelfAttention(nn.Module):
    """
    Self-attention module for feature maps.
    
    Args:
        channels (int): Number of input channels.
        dimensions (int): Dimensionality of the data (1, 2 or 3).
    """
    def __init__(self, channels, dimensions=2):
        super().__init__()
        self.channels = channels
        self.query = nn.Conv2d(channels, channels // 8, kernel_size=1) if dimensions == 2 else \
                    nn.Conv3d(channels, channels // 8, kernel_size=1)
        self.key = nn.Conv2d(channels, channels // 8, kernel_size=1) if dimensions == 2 else \
                    nn.Conv3d(channels, channels // 8, kernel_size=1)
        self.value = nn.Conv2d(channels, channels, kernel_size=1) if dimensions == 2 else \
                    nn.Conv3d(channels, channels, kernel_size=1)
        self.gamma = nn.Parameter(torch.zeros(1))
                
    def forward(self, x):
        batch_size, channels, *spatial_dims = x.size()
        
        # Reshape for attention computation
        proj_query = self.query(x).view(batch_size, -1, spatial_dims[0] * spatial_dims[1])
        proj_key = self.key(x).view(batch_size, -1, spatial_dims[0] * spatial_dims[1])
        proj_value = self.value(x).view(batch_size, -1, spatial_dims[0] * spatial_dims[1])
        
        # Transpose for matrix multiplication
        proj_query = proj_query.permute(0, 2, 1)  # B, HW, C
        
        # Calculate attention map
        attention_map = torch.bmm(proj_query, proj_key)  # B, HW, HW
        attention_map = F.softmax(attention_map, dim=-1)
        
        # Apply attention to values
        out = torch.bmm(proj_value, attention_map.permute(0, 2, 1))
        out = out.view(batch_size, channels, *spatial_dims)
        
        # Apply residual connection with learnable weight
        out = self.gamma * out + x
        
        return out


class Conv(nn.Module):
    """
    Convolutional layer for 1, 2, or 3D data.

    Same arugments as nn.Convxd but dimensionality of data is
    passed as argument.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        device=None,
        dtype=None,
        dimensions=2,
    ):
        super().__init__()
        self.args = {
            "in_channels": in_channels,
            "out_channels": out_channels,
            "kernel_size": kernel_size,
            "stride": stride,
            "padding": padding,
            "dilation": dilation,
            "groups": groups,
            "bias": bias,
            "padding_mode": padding_mode,
            "device": device,
            "dtype": dtype,
        }
        conv = getattr(nn, f"Conv{dimensions}d")
        self.conv = conv(**self.args)

    def forward(self, x):
        return self.conv(x)


class ResBlockWithResampling(nn.Module):
    """
    Residual Block with Resampling.

    Args:
        c_in (int): Number of input channels.
        c_out (int): Number of output channels.
        resample (str): Resampling method. Can be "up", "down", or None.
        res_block_kernel (int): Kernel size for the residual block.
        groups (int): Number of groups for grouped convolution.
        gated (bool): Whether to use gated activation.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5). 
            See VDVAE, Child 2020.
        dimensions (int): Dimensionality of the data (1, 2 or 3).
        attention (bool): Whether to use self-attention.

    Attributes:
        pre_conv (nn.Module): Pre-convolutional layer.
        res_block (ResidualBlock): Residual block layer.
        attention (nn.Module): Self-attention module if enabled.
    """

    def __init__(
        self,
        c_in,
        c_out,
        resample=None,
        res_block_kernel=3,
        groups=1,
        gated=True,
        scale_initialisation=False,
        dimensions=2,
        attention=False
    ):
        super().__init__()
        assert resample in ["up", "down", None]
        self.use_attention = attention

        if resample == "up":
            self.pre_conv = nn.Sequential(
                nn.Upsample(scale_factor=2, mode="nearest"),
                Conv(c_in, c_out, 1, groups=groups, dimensions=dimensions),
            )
        elif resample == "down":
            self.pre_conv = Conv(
                c_in,
                c_out,
                kernel_size=3,
                stride=2,
                padding=1,
                padding_mode="reflect",
                groups=groups,
                dimensions=dimensions,
            )
        elif c_in != c_out:
            self.pre_conv = Conv(c_in, c_out, 1, groups=groups, dimensions=dimensions)
        else:
            self.pre_conv = nn.Identity()

        self.res_block = ResidualBlock(
            channels=c_out,
            kernel_size=res_block_kernel,
            groups=groups,
            gated=gated,
            scale_initialisation=scale_initialisation,
            dimensions=dimensions,
        )
        
        if self.use_attention:
            self.attention = SelfAttention(c_out, dimensions=dimensions)

    def forward(self, x):
        x = self.pre_conv(x)
        x = self.res_block(x)
        if self.use_attention:
            x = self.attention(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(
        self,
        channels,
        kernel_size=3,
        groups=1,
        gated=True,
        scale_initialisation=False,
        dimensions=2,
    ):
        super().__init__()
        self.channels = channels
        assert kernel_size % 2 == 1
        self.pad = kernel_size // 2
        self.kernel_size = kernel_size
        self.groups = groups
        self.gated = gated
        if scale_initialisation:
            self.rescale = channels ** 0.5
        else:
            self.rescale = 1.0

        BatchNorm = getattr(nn, f"BatchNorm{dimensions}d")

        self.block = nn.Sequential()
        for _ in range(2):
            self.block.append(BatchNorm(self.channels))
            self.block.append(nn.ReLU(inplace=True))
            conv = Conv(
                self.channels,
                self.channels,
                self.kernel_size,
                padding=self.pad,
                groups=self.groups,
                dimensions=dimensions,
            )
            self.block.append(conv)
        if self.gated:
            self.block.append(GateLayer(self.channels, 1, dimensions=dimensions))

    def forward(self, x):
        out = self.block(x) + x
        out = out / self.rescale
        return out


class GateLayer(nn.Module):
    def __init__(self, channels, kernel_size, dimensions=2):
        super().__init__()
        assert kernel_size % 2 == 1
        pad = kernel_size // 2
        self.conv = Conv(channels, 2 * channels, kernel_size, padding=pad, dimensions=dimensions)
        self.nonlin = nn.Tanh()

    def forward(self, x):
        x = self.conv(x)
        x, gate = x[:, 0::2], x[:, 1::2]
        x = self.nonlin(x)
        gate = torch.sigmoid(gate)
        return x * gate


class BottomUpLayer(nn.Module):
    """
    Module that consists of multiple residual blocks.
    Each residual block can optionally perform downsampling.

    Args:
        n_res_blocks (int): The number of residual blocks in the layer.
        n_filters (int): The number of filters in each residual block.
        downsampling_steps (int, optional): The number of downsampling steps to perform. Defaults to 0.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5). 
            See VDVAE, Child 2020.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
        attention (bool): Whether to use self-attention.
    """

    def __init__(
        self,
        n_res_blocks,
        n_filters,
        downsampling_steps=0,
        scale_initialisation=False,
        dimensions=2,
        attention=False
    ):
        super().__init__()

        self.bu_blocks = nn.ModuleList()
        for i in range(n_res_blocks):
            resample = None
            if downsampling_steps > 0:
                resample = "down"
                downsampling_steps -= 1
            
            # Add attention to deeper layers
            use_attention = attention and (i == n_res_blocks - 1)
            
            self.bu_blocks.append(
                ResBlockWithResampling(
                    c_in=n_filters,
                    c_out=n_filters,
                    resample=resample,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                    attention=use_attention
                )
            )

    def forward(self, x):
        for block in self.bu_blocks:
            x = block(x)
        return x


class MergeLayer(nn.Module):
    """
    A module that merges two input tensors and applies a convolutional layer followed by a residual block.

    Args:
        channels (int or list[int]): The number of input channels for the convolutional layer and the residual block.
            If an integer is provided, it will be used for all three channels. If a list of integers is provided,
            it should have a length of 3, representing the number of channels for each input.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5). 
            See VDVAE, Child 2020.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
        attention (bool): Whether to use self-attention.
    """

    def __init__(self, channels, scale_initialisation=False, dimensions=2, attention=False):
        super().__init__()
        try:
            iter(channels)
        except TypeError:  # it is not iterable
            channels = [channels] * 3
        else:  # it is iterable
            if len(channels) == 1:
                channels = [channels[0]] * 3
        assert len(channels) == 3

        self.use_attention = attention
        
        self.conv = Conv(channels[0] + channels[1], channels[2], 1, dimensions=dimensions)
        self.res_block = ResidualBlock(
            channels[2],
            scale_initialisation=scale_initialisation,
            dimensions=dimensions,
        )
        
        if self.use_attention:
            self.attention = SelfAttention(channels[2], dimensions=dimensions)

    def forward(self, x, y):
        # Concatenate along channel dimension
        combined = torch.cat([x, y], dim=1)
        out = self.conv(combined)
        out = self.res_block(out)
        
        if self.use_attention:
            out = self.attention(out)
            
        return out


class TopDownLayer(nn.Module):
    """
    Module that consists of multiple residual blocks.
    Each residual block can optionally perform upsampling.

    Merges a bottom-up skip connection and optionally a skip connection from the previous layer.

    Args:
        n_res_blocks (int): The number of residual blocks in the layer.
        n_filters (int): The number of filters in each residual block.
        is_top_layer (bool): Whether the layer is the top layer.
        upsampling_steps (int, optional): The number of downsampling steps to perform. Defaults to 0.
        skip (bool, optional): Whether to use a skip connection from the previous layer. Defaults to False.
        scale_initialisation (bool): For stability, scale the last layer in the residual block by 1/(depth**0.5). 
            See VDVAE, Child 2020.
        dimensions (int): Dimensionality of the data (1, 2 or 3)
        attention (bool): Whether to use self-attention.
    """

    def __init__(
        self,
        n_res_blocks,
        n_filters,
        is_top_layer=False,
        upsampling_steps=None,
        skip=False,
        scale_initialisation=False,
        dimensions=2,
        attention=False
    ):
        super().__init__()

        self.is_top_layer = is_top_layer
        self.skip = skip

        self.blocks = nn.ModuleList()
        for i in range(n_res_blocks):
            resample = None
            if upsampling_steps > 0:
                resample = "up"
                upsampling_steps -= 1
            
            # Add attention to deeper layers
            use_attention = attention and (i == 0)
            
            self.blocks.append(
                ResBlockWithResampling(
                    n_filters,
                    n_filters,
                    resample=resample,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                    attention=use_attention
                )
            )

        if not is_top_layer:
            self.merge = MergeLayer(
                channels=n_filters,
                scale_initialisation=scale_initialisation,
                dimensions=dimensions,
                attention=attention
            )

            if skip:
                self.skip_connection_merger = MergeLayer(
                    channels=n_filters,
                    scale_initialisation=scale_initialisation,
                    dimensions=dimensions,
                    attention=attention
                )

    def forward(
        self,
        input_=None,
        skip_connection_input=None,
        bu_value=None,
    ):
        if self.is_top_layer:
            output = bu_value
        else:
            output = self.merge(bu_value, input_)

        # Skip connection from previous layer
        if self.skip and not self.is_top_layer:
            output = self.skip_connection_merger(output, skip_connection_input)

        # Process through blocks
        for block in self.blocks:
            output = block(output)

        return output