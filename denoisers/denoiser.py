from collections import OrderedDict
import math
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""
    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)

class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma

class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = LayerNorm,
            idx: int = 12,
    ):
        super().__init__()
        self.idx = idx
        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        x = x + self.ls_1(self.attn(self.ln_1(x), self.ln_1(x), self.ln_1(x), attn_mask=attn_mask)[0])
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x

class TransformerEncoder(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: nn.Module = nn.GELU,
            norm_layer: nn.Module = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(width, heads, mlp_ratio, ls_init_value, act_layer, norm_layer, idx)
            for idx in range(layers)
        ])

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        for r in self.resblocks:
            x = r(x, attn_mask=attn_mask)
        return x