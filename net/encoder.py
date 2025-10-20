from net.modules import *
import torch
import torch.nn as nn
# from timm.models.registry import register_model
# import math
# from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
# from timm.models._builder import resolve_pretrained_cfg
# try:
#     from timm.models._builder import _update_default_kwargs as update_args
# except:
#     from timm.models._builder import _update_default_model_kwargs as update_args
# from timm.models.vision_transformer import Mlp, PatchEmbed
# from timm.models.layers import DropPath, trunc_normal_
# from timm.models.registry import register_model
# import torch.nn.functional as F
# from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
# from einops import rearrange, repeat
# from .registry import register_pip_model
# from pathlib import Path
# !/usr/bin/env python3

# Copyright (c) 2024, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
from timm.models.registry import register_model
import math
from timm.models.layers import trunc_normal_, DropPath, LayerNorm2d
from timm.models._builder import resolve_pretrained_cfg

try:
    from timm.models._builder import _update_default_kwargs as update_args
except:
    from timm.models._builder import _update_default_model_kwargs as update_args
from timm.models.vision_transformer import Mlp, PatchEmbed
from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn
from einops import rearrange, repeat
from net.registry import register_pip_model
from pathlib import Path


def _cfg(url='', **kwargs):
    return {'url': url,
            'num_classes': 1000,
            'input_size': (3, 224, 224),
            'pool_size': None,
            'crop_pct': 0.875,
            'interpolation': 'bicubic',
            'fixed_input_size': True,
            'mean': (0.485, 0.456, 0.406),
            'std': (0.229, 0.224, 0.225),
            **kwargs
            }


default_cfgs = {
    'mamba_vision_T': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-T-1K/resolve/main/mambavision_tiny_1k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_T2': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-T2-1K/resolve/main/mambavision_tiny2_1k.pth.tar',
        crop_pct=0.98,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_S': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-S-1K/resolve/main/mambavision_small_1k.pth.tar',
        crop_pct=0.93,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_B': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-B-1K/resolve/main/mambavision_base_1k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_L': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-L-1K/resolve/main/mambavision_large_1k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center'),
    'mamba_vision_L2': _cfg(
        url='https://huggingface.co/nvidia/MambaVision-L2-1K/resolve/main/mambavision_large2_1k.pth.tar',
        crop_pct=1.0,
        input_size=(3, 224, 224),
        crop_mode='center')
}


def window_partition(x, window_size):
    """
    Args:
        x: (B, C, H, W)
        window_size: window size
        h_w: Height of window
        w_w: Width of window
    Returns:
        local window features (num_windows*B, window_size*window_size, C)
    """
    B, C, H, W = x.shape
    x = x.view(B, C, H // window_size, window_size, W // window_size, window_size)
    windows = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, window_size * window_size, C)
    return windows


def window_reverse(windows, window_size, H, W):
    """
    Args:
        windows: local window features (num_windows*B, window_size, window_size, C)
        window_size: Window size
        H: Height of image
        W: Width of image
    Returns:
        x: (B, C, H, W)
    """
    B = int(windows.shape[0] / (H * W / window_size / window_size))
    x = windows.reshape(B, H // window_size, W // window_size, window_size, window_size, -1)
    x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, windows.shape[2], H, W)
    return x


def _load_state_dict(module, state_dict, strict=False, logger=None):
    """Load state_dict to a module.

    This method is modified from :meth:`torch.nn.Module.load_state_dict`.
    Default value for ``strict`` is set to ``False`` and the message for
    param mismatch will be shown even if strict is False.

    Args:
        module (Module): Module that receives the state_dict.
        state_dict (OrderedDict): Weights.
        strict (bool): whether to strictly enforce that the keys
            in :attr:`state_dict` match the keys returned by this module's
            :meth:`~torch.nn.Module.state_dict` function. Default: ``False``.
        logger (:obj:`logging.Logger`, optional): Logger to log the error
            message. If not specified, print function will be used.
    """
    unexpected_keys = []
    all_missing_keys = []
    err_msg = []

    metadata = getattr(state_dict, '_metadata', None)
    state_dict = state_dict.copy()
    if metadata is not None:
        state_dict._metadata = metadata

    def load(module, prefix=''):
        local_metadata = {} if metadata is None else metadata.get(
            prefix[:-1], {})
        module._load_from_state_dict(state_dict, prefix, local_metadata, True,
                                     all_missing_keys, unexpected_keys,
                                     err_msg)
        for name, child in module._modules.items():
            if child is not None:
                load(child, prefix + name + '.')

    load(module)
    load = None
    missing_keys = [
        key for key in all_missing_keys if 'num_batches_tracked' not in key
    ]

    if unexpected_keys:
        err_msg.append('unexpected key in source '
                       f'state_dict: {", ".join(unexpected_keys)}\n')
    if missing_keys:
        err_msg.append(
            f'missing keys in source state_dict: {", ".join(missing_keys)}\n')

    if len(err_msg) > 0:
        err_msg.insert(
            0, 'The model and loaded state dict do not match exactly\n')
        err_msg = '\n'.join(err_msg)
        if strict:
            raise RuntimeError(err_msg)
        elif logger is not None:
            logger.warning(err_msg)
        else:
            print(err_msg)


def _load_checkpoint(model,
                     filename,
                     map_location='cpu',
                     strict=False,
                     logger=None):
    """Load checkpoint from a file or URI.

    Args:
        model (Module): Module to load checkpoint.
        filename (str): Accept local filepath, URL, ``torchvision://xxx``,
            ``open-mmlab://xxx``. Please refer to ``docs/model_zoo.md`` for
            details.
        map_location (str): Same as :func:`torch.load`.
        strict (bool): Whether to allow different params for the model and
            checkpoint.
        logger (:mod:`logging.Logger` or None): The logger for error message.

    Returns:
        dict or OrderedDict: The loaded checkpoint.
    """
    checkpoint = torch.load(filename, map_location=map_location)
    if not isinstance(checkpoint, dict):
        raise RuntimeError(
            f'No state_dict found in checkpoint file {filename}')
    if 'state_dict' in checkpoint:
        state_dict = checkpoint['state_dict']
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
    else:
        state_dict = checkpoint
    if list(state_dict.keys())[0].startswith('module.'):
        state_dict = {k[7:]: v for k, v in state_dict.items()}

    if sorted(list(state_dict.keys()))[0].startswith('encoder'):
        state_dict = {k.replace('encoder.', ''): v for k, v in state_dict.items() if k.startswith('encoder.')}

    _load_state_dict(model, state_dict, strict, logger)
    return checkpoint


class Downsample(nn.Module):
    """
    Down-sampling block"
    """

    def __init__(self,
                 dim,
                 keep_dim=False,
                 ):
        """
        Args:
            dim: feature size dimension.
            norm_layer: normalization layer.
            keep_dim: bool argument for maintaining the resolution.
        """

        super().__init__()
        if keep_dim:
            dim_out = dim
        else:
            dim_out = 2 * dim
        self.reduction = nn.Sequential(
            nn.Conv2d(dim, dim_out, 3, 2, 1, bias=False),
        )

    def forward(self, x):
        x = self.reduction(x)
        return x


class PatchEmbed(nn.Module):
    """
    Patch embedding block"
    """

    def __init__(self, in_chans=3, in_dim=64, dim=96):
        """
        Args:
            in_chans: number of input channels.
            dim: feature size dimension.
        """
        # in_dim = 1
        super().__init__()
        self.proj = nn.Identity()
        self.conv_down = nn.Sequential(
            nn.Conv2d(in_chans, in_dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(in_dim, eps=1e-4),
            nn.ReLU(),
            nn.Conv2d(in_dim, dim, 3, 2, 1, bias=False),
            nn.BatchNorm2d(dim, eps=1e-4),
            nn.ReLU()
        )

    def forward(self, x):
        x = self.proj(x)
        x = self.conv_down(x)
        return x


class ConvBlock(nn.Module):

    def __init__(self, dim,
                 drop_path=0.,
                 layer_scale=None,
                 kernel_size=3):
        super().__init__()

        self.conv1 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm1 = nn.BatchNorm2d(dim, eps=1e-5)
        self.act1 = nn.GELU(approximate='tanh')
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=kernel_size, stride=1, padding=1)
        self.norm2 = nn.BatchNorm2d(dim, eps=1e-5)
        self.layer_scale = layer_scale
        if layer_scale is not None and type(layer_scale) in [int, float]:
            self.gamma = nn.Parameter(layer_scale * torch.ones(dim))
            self.layer_scale = True
        else:
            self.layer_scale = False
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.norm2(x)
        if self.layer_scale:
            x = x * self.gamma.view(1, -1, 1, 1)
        x = input + self.drop_path(x)
        return x


class MambaVisionMixer(nn.Module):
    def __init__(
            self,
            d_model,
            d_state=16,
            d_conv=4,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            conv_bias=True,
            bias=False,
            use_fast_path=True,
            layer_idx=None,
            device=None,
            dtype=None,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank
        self.use_fast_path = use_fast_path
        self.layer_idx = layer_idx
        self.in_proj = nn.Linear(self.d_model, self.d_inner, bias=bias, **factory_kwargs)
        self.x_proj = nn.Linear(
            self.d_inner // 2, self.dt_rank + self.d_state * 2, bias=False, **factory_kwargs
        )
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner // 2, bias=True, **factory_kwargs)
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(self.dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(self.dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError
        dt = torch.exp(
            torch.rand(self.d_inner // 2, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            self.dt_proj.bias.copy_(inv_dt)
        self.dt_proj.bias._no_reinit = True
        A = repeat(
            torch.arange(1, self.d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=self.d_inner // 2,
        ).contiguous()
        A_log = torch.log(A)
        self.A_log = nn.Parameter(A_log)
        self.A_log._no_weight_decay = True
        self.D = nn.Parameter(torch.ones(self.d_inner // 2, device=device))
        self.D._no_weight_decay = True
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.conv1d_x = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )
        self.conv1d_z = nn.Conv1d(
            in_channels=self.d_inner // 2,
            out_channels=self.d_inner // 2,
            bias=conv_bias // 2,
            kernel_size=d_conv,
            groups=self.d_inner // 2,
            **factory_kwargs,
        )

    def forward(self, hidden_states):
        """
        hidden_states: (B, L, D)
        Returns: same shape as hidden_states
        """
        _, seqlen, _ = hidden_states.shape
        xz = self.in_proj(hidden_states)
        xz = rearrange(xz, "b l d -> b d l")
        x, z = xz.chunk(2, dim=1)
        A = -torch.exp(self.A_log.float())
        x = F.silu(F.conv1d(input=x, weight=self.conv1d_x.weight, bias=self.conv1d_x.bias, padding='same',
                            groups=self.d_inner // 2))
        z = F.silu(F.conv1d(input=z, weight=self.conv1d_z.weight, bias=self.conv1d_z.bias, padding='same',
                            groups=self.d_inner // 2))
        x_dbl = self.x_proj(rearrange(x, "b d l -> (b l) d"))
        dt, B, C = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = rearrange(self.dt_proj(dt), "(b l) d -> b d l", l=seqlen)
        B = rearrange(B, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        C = rearrange(C, "(b l) dstate -> b dstate l", l=seqlen).contiguous()
        y = selective_scan_fn(x,
                              dt,
                              A,
                              B,
                              C,
                              self.D.float(),
                              z=None,
                              delta_bias=self.dt_proj.bias.float(),
                              delta_softplus=True,
                              return_last_state=None)

        y = torch.cat([y, z], dim=1)
        y = rearrange(y, "b d l -> b l d")
        out = self.out_proj(y)
        return out


# class Attention(nn.Module):
#
#     def __init__(
#             self,
#             dim,
#             num_heads=8,
#             qkv_bias=False,
#             qk_norm=False,
#             attn_drop=0.,
#             proj_drop=0.,
#             norm_layer=nn.LayerNorm,
#     ):
#         super().__init__()
#         assert dim % num_heads == 0
#         self.num_heads = num_heads
#         self.head_dim = dim // num_heads
#         self.scale = self.head_dim ** -0.5
#         self.fused_attn = True
#
#         self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
#         self.q_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.k_norm = norm_layer(self.head_dim) if qk_norm else nn.Identity()
#         self.attn_drop = nn.Dropout(attn_drop)
#         self.proj = nn.Linear(dim, dim)
#         self.proj_drop = nn.Dropout(proj_drop)
#
#     def forward(self, x):
#         B, N, C = x.shape
#         qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
#         q, k, v = qkv.unbind(0)
#         q, k = self.q_norm(q), self.k_norm(k)
#
#         if self.fused_attn:
#             x = F.scaled_dot_product_attention(
#                 q, k, v,
#                 dropout_p=self.attn_drop.p,
#             )
#         else:
#             q = q * self.scale
#             attn = q @ k.transpose(-2, -1)
#             attn = attn.softmax(dim=-1)
#             attn = self.attn_drop(attn)
#             x = attn @ v
#
#         x = x.transpose(1, 2).reshape(B, N, C)
#         x = self.proj(x)
#         x = self.proj_drop(x)
#         return x
#
#
# class Block(nn.Module):
#     def __init__(self,
#                  dim,
#                  num_heads,
#                  counter,
#                  transformer_blocks,
#                  mlp_ratio=4.,
#                  qkv_bias=False,
#                  qk_scale=False,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  act_layer=nn.GELU,
#                  norm_layer=nn.LayerNorm,
#                  Mlp_block=Mlp,
#                  layer_scale=None,
#                  ):
#         super().__init__()
#         self.norm1 = norm_layer(dim)
#         if counter in transformer_blocks:
#             self.mixer = Attention(
#                 dim,
#                 num_heads=num_heads,
#                 qkv_bias=qkv_bias,
#                 qk_norm=qk_scale,
#                 attn_drop=attn_drop,
#                 proj_drop=drop,
#                 norm_layer=norm_layer,
#             )
#         else:
#             self.mixer = MambaVisionMixer(d_model=dim,
#                                           d_state=8,
#                                           d_conv=3,
#                                           expand=1
#                                           )
#
#         self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
#         self.norm2 = norm_layer(dim)
#         mlp_hidden_dim = int(dim * mlp_ratio)
#         self.mlp = Mlp_block(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
#         use_layer_scale = layer_scale is not None and type(layer_scale) in [int, float]
#         self.gamma_1 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
#         self.gamma_2 = nn.Parameter(layer_scale * torch.ones(dim)) if use_layer_scale else 1
#
#     def forward(self, x):
#         x = x + self.drop_path(self.gamma_1 * self.mixer(self.norm1(x)))
#         x = x + self.drop_path(self.gamma_2 * self.mlp(self.norm2(x)))
#         return x
#
#
# class MambaVisionLayer(nn.Module):
#     """
#     MambaVision layer"
#     """
#
#     def __init__(self,
#                  dim,
#                  depth,
#                  num_heads,
#                  window_size,
#                  conv=False,
#                  downsample=True,
#                  mlp_ratio=4.,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop=0.,
#                  attn_drop=0.,
#                  drop_path=0.,
#                  layer_scale=None,
#                  layer_scale_conv=None,
#                  transformer_blocks=[],
#                  ):
#         """
#         Args:
#             dim: feature size dimension.
#             depth: number of layers in each stage.
#             window_size: window size in each stage.
#             conv: bool argument for conv stage flag.
#             downsample: bool argument for down-sampling.
#             mlp_ratio: MLP ratio.
#             num_heads: number of heads in each stage.
#             qkv_bias: bool argument for query, key, value learnable bias.
#             qk_scale: bool argument to scaling query, key.
#             drop: dropout rate.
#             attn_drop: attention dropout rate.
#             drop_path: drop path rate.
#             norm_layer: normalization layer.
#             layer_scale: layer scaling coefficient.
#             layer_scale_conv: conv layer scaling coefficient.
#             transformer_blocks: list of transformer blocks.
#         """
#
#         super().__init__()
#         self.conv = conv
#         self.transformer_block = False
#         if conv:
#             self.blocks = nn.ModuleList([ConvBlock(dim=dim,
#                                                    drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                                    layer_scale=layer_scale_conv)
#                                          for i in range(depth)])
#             self.transformer_block = False
#         else:
#             self.blocks = nn.ModuleList([Block(dim=dim,
#                                                counter=i,
#                                                transformer_blocks=transformer_blocks,
#                                                num_heads=num_heads,
#                                                mlp_ratio=mlp_ratio,
#                                                qkv_bias=qkv_bias,
#                                                qk_scale=qk_scale,
#                                                drop=drop,
#                                                attn_drop=attn_drop,
#                                                drop_path=drop_path[i] if isinstance(drop_path, list) else drop_path,
#                                                layer_scale=layer_scale)
#                                          for i in range(depth)])
#             self.transformer_block = True
#
#         self.downsample = None if not downsample else Downsample(dim=dim)
#         self.do_gt = False
#         self.window_size = window_size
#
#     def forward(self, x):
#         _, _, H, W = x.shape
#
#         if self.transformer_block:
#             pad_r = (self.window_size - W % self.window_size) % self.window_size
#             pad_b = (self.window_size - H % self.window_size) % self.window_size
#             if pad_r > 0 or pad_b > 0:
#                 x = torch.nn.functional.pad(x, (0, pad_r, 0, pad_b))
#                 _, _, Hp, Wp = x.shape
#             else:
#                 Hp, Wp = H, W
#             x = window_partition(x, self.window_size)
#
#         for _, blk in enumerate(self.blocks):
#             x = blk(x)
#         if self.transformer_block:
#             x = window_reverse(x, self.window_size, Hp, Wp)
#             if pad_r > 0 or pad_b > 0:
#                 x = x[:, :, :H, :W].contiguous()
#         if self.downsample is None:
#             return x
#         return self.downsample(x)
#
#
# class MambaVision(nn.Module):
#     """
#     MambaVision,
#     """
#
#     def __init__(self,
#                  dim,
#                  in_dim,
#                  depths,
#                  window_size,
#                  mlp_ratio,
#                  num_heads,
#                  drop_path_rate=0.2,
#                  in_chans=3,
#                  num_classes=1000,
#                  qkv_bias=True,
#                  qk_scale=None,
#                  drop_rate=0.,
#                  attn_drop_rate=0.,
#                  layer_scale=None,
#                  layer_scale_conv=None,
#                  **kwargs):
#         """
#         Args:
#             dim: feature size dimension.
#             depths: number of layers in each stage.
#             window_size: window size in each stage.
#             mlp_ratio: MLP ratio.
#             num_heads: number of heads in each stage.
#             drop_path_rate: drop path rate.
#             in_chans: number of input channels.
#             num_classes: number of classes.
#             qkv_bias: bool argument for query, key, value learnable bias.
#             qk_scale: bool argument to scaling query, key.
#             drop_rate: dropout rate.
#             attn_drop_rate: attention dropout rate.
#             norm_layer: normalization layer.
#             layer_scale: layer scaling coefficient.
#             layer_scale_conv: conv layer scaling coefficient.
#         """
#         super().__init__()
#         num_features = int(dim * 2 ** (len(depths) - 1))
#         self.num_classes = num_classes
#         self.patch_embed = PatchEmbed(in_chans=in_chans, in_dim=in_dim, dim=dim)
#         dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
#         self.levels = nn.ModuleList()
#         for i in range(len(depths)):
#             conv = True if (i == 0 or i == 1) else False
#             level = MambaVisionLayer(dim=int(dim * 2 ** i),
#                                      depth=depths[i],
#                                      num_heads=num_heads[i],
#                                      window_size=window_size[i],
#                                      mlp_ratio=mlp_ratio,
#                                      qkv_bias=qkv_bias,
#                                      qk_scale=qk_scale,
#                                      conv=conv,
#                                      drop=drop_rate,
#                                      attn_drop=attn_drop_rate,
#                                      drop_path=dpr[sum(depths[:i]):sum(depths[:i + 1])],
#                                      downsample=(i < 3),
#                                      layer_scale=layer_scale,
#                                      layer_scale_conv=layer_scale_conv,
#                                      transformer_blocks=list(range(depths[i] // 2 + 1, depths[i])) if depths[
#                                                                                                           i] % 2 != 0 else list(
#                                          range(depths[i] // 2, depths[i])),
#                                      )
#             self.levels.append(level)
#         self.norm = nn.BatchNorm2d(num_features)
#         self.avgpool = nn.AdaptiveAvgPool2d(1)
#         self.head = nn.Linear(num_features, num_classes) if num_classes > 0 else nn.Identity()
#         self.apply(self._init_weights)
#
#     def _init_weights(self, m):
#         if isinstance(m, nn.Linear):
#             trunc_normal_(m.weight, std=.02)
#             if isinstance(m, nn.Linear) and m.bias is not None:
#                 nn.init.constant_(m.bias, 0)
#         elif isinstance(m, nn.LayerNorm):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, LayerNorm2d):
#             nn.init.constant_(m.bias, 0)
#             nn.init.constant_(m.weight, 1.0)
#         elif isinstance(m, nn.BatchNorm2d):
#             nn.init.ones_(m.weight)
#             nn.init.zeros_(m.bias)
#
#     @torch.jit.ignore
#     def no_weight_decay_keywords(self):
#         return {'rpb'}
#
#     def forward_features(self, x):
#         x = self.patch_embed(x)
#         for level in self.levels:
#             x = level(x)
#         x = self.norm(x)
#         x = self.avgpool(x)
#         x = torch.flatten(x, 1)
#         return x
#
#     def forward(self, x):
#         x = self.forward_features(x)
#         x = self.head(x)
#         return x
#
#     def _load_state_dict(self,
#                          pretrained,
#                          strict: bool = False):
#         _load_checkpoint(self,
#                          pretrained,
#                          strict=strict)
#
#
# @register_pip_model
# @register_model
# def mamba_vision_T(pretrained=False, **kwargs):
#     model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T.pth.tar")
#     depths = kwargs.pop("depths", [1, 3, 8, 4])
#     num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
#     window_size = kwargs.pop("window_size", [8, 8, 14, 7])
#     dim = kwargs.pop("dim", 80)
#     in_dim = kwargs.pop("in_dim", 32)
#     mlp_ratio = kwargs.pop("mlp_ratio", 4)
#     resolution = kwargs.pop("resolution", 224)
#     drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
#     pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T').to_dict()
#     update_args(pretrained_cfg, kwargs, kwargs_filter=None)
#     model = MambaVision(depths=[1, 3, 8, 4],
#                         num_heads=[2, 4, 8, 16],
#                         window_size=[8, 8, 14, 7],
#                         dim=80,
#                         in_dim=32,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.2,
#                         **kwargs)
#     model.pretrained_cfg = pretrained_cfg
#     model.default_cfg = model.pretrained_cfg
#     if pretrained:
#         if not Path(model_path).is_file():
#             url = model.default_cfg['url']
#             torch.hub.download_url_to_file(url=url, dst=model_path)
#         model._load_state_dict(model_path)
#     return model
#
#
# @register_pip_model
# @register_model
# def mamba_vision_T2(pretrained=False, **kwargs):
#     model_path = kwargs.pop("model_path", "/tmp/mamba_vision_T2.pth.tar")
#     depths = kwargs.pop("depths", [1, 3, 11, 4])
#     num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
#     window_size = kwargs.pop("window_size", [8, 8, 14, 7])
#     dim = kwargs.pop("dim", 80)
#     in_dim = kwargs.pop("in_dim", 32)
#     mlp_ratio = kwargs.pop("mlp_ratio", 4)
#     resolution = kwargs.pop("resolution", 224)
#     drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
#     pretrained_cfg = resolve_pretrained_cfg('mamba_vision_T2').to_dict()
#     update_args(pretrained_cfg, kwargs, kwargs_filter=None)
#     model = MambaVision(depths=[1, 3, 11, 4],
#                         num_heads=[2, 4, 8, 16],
#                         window_size=[8, 8, 14, 7],
#                         dim=80,
#                         in_dim=32,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.2,
#                         **kwargs)
#     model.pretrained_cfg = pretrained_cfg
#     model.default_cfg = model.pretrained_cfg
#     if pretrained:
#         if not Path(model_path).is_file():
#             url = model.default_cfg['url']
#             torch.hub.download_url_to_file(url=url, dst=model_path)
#         model._load_state_dict(model_path)
#     return model
#
#
# @register_pip_model
# @register_model
# def mamba_vision_S(pretrained=False, **kwargs):
#     model_path = kwargs.pop("model_path", "/tmp/mamba_vision_S.pth.tar")
#     depths = kwargs.pop("depths", [3, 3, 7, 5])
#     num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
#     window_size = kwargs.pop("window_size", [8, 8, 14, 7])
#     dim = kwargs.pop("dim", 96)
#     in_dim = kwargs.pop("in_dim", 64)
#     mlp_ratio = kwargs.pop("mlp_ratio", 4)
#     resolution = kwargs.pop("resolution", 224)
#     drop_path_rate = kwargs.pop("drop_path_rate", 0.2)
#     pretrained_cfg = resolve_pretrained_cfg('mamba_vision_S').to_dict()
#     update_args(pretrained_cfg, kwargs, kwargs_filter=None)
#     model = MambaVision(depths=[3, 3, 7, 5],
#                         num_heads=[2, 4, 8, 16],
#                         window_size=[8, 8, 14, 7],
#                         dim=96,
#                         in_dim=64,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.2,
#                         **kwargs)
#     model.pretrained_cfg = pretrained_cfg
#     model.default_cfg = model.pretrained_cfg
#     if pretrained:
#         if not Path(model_path).is_file():
#             url = model.default_cfg['url']
#             torch.hub.download_url_to_file(url=url, dst=model_path)
#         model._load_state_dict(model_path)
#     return model
#
#
# @register_pip_model
# @register_model
# def mamba_vision_B(pretrained=False, **kwargs):
#     model_path = kwargs.pop("model_path", "/tmp/mamba_vision_B.pth.tar")
#     depths = kwargs.pop("depths", [3, 3, 10, 5])
#     num_heads = kwargs.pop("num_heads", [2, 4, 8, 16])
#     window_size = kwargs.pop("window_size", [8, 8, 14, 7])
#     dim = kwargs.pop("dim", 128)
#     in_dim = kwargs.pop("in_dim", 64)
#     mlp_ratio = kwargs.pop("mlp_ratio", 4)
#     resolution = kwargs.pop("resolution", 224)
#     drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
#     layer_scale = kwargs.pop("layer_scale", 1e-5)
#     pretrained_cfg = resolve_pretrained_cfg('mamba_vision_B').to_dict()
#     update_args(pretrained_cfg, kwargs, kwargs_filter=None)
#     model = MambaVision(depths=[3, 3, 10, 5],
#                         num_heads=[2, 4, 8, 16],
#                         window_size=[8, 8, 14, 7],
#                         dim=128,
#                         in_dim=64,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.3,
#                         layer_scale=1e-5,
#                         layer_scale_conv=None,
#                         **kwargs)
#     model.pretrained_cfg = pretrained_cfg
#     model.default_cfg = model.pretrained_cfg
#     if pretrained:
#         if not Path(model_path).is_file():
#             url = model.default_cfg['url']
#             torch.hub.download_url_to_file(url=url, dst=model_path)
#         model._load_state_dict(model_path)
#     return model
#
#
# @register_pip_model
# @register_model
# def mamba_vision_L(pretrained=False, **kwargs):
#     model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L.pth.tar")
#     depths = kwargs.pop("depths", [3, 3, 10, 5])
#     num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
#     window_size = kwargs.pop("window_size", [8, 8, 14, 7])
#     dim = kwargs.pop("dim", 196)
#     in_dim = kwargs.pop("in_dim", 64)
#     mlp_ratio = kwargs.pop("mlp_ratio", 4)
#     resolution = kwargs.pop("resolution", 224)
#     drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
#     layer_scale = kwargs.pop("layer_scale", 1e-5)
#     pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L').to_dict()
#     update_args(pretrained_cfg, kwargs, kwargs_filter=None)
#     model = MambaVision(depths=[3, 3, 10, 5],
#                         num_heads=[4, 8, 16, 32],
#                         window_size=[8, 8, 14, 7],
#                         dim=196,
#                         in_dim=64,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.3,
#                         layer_scale=1e-5,
#                         layer_scale_conv=None,
#                         **kwargs)
#     model.pretrained_cfg = pretrained_cfg
#     model.default_cfg = model.pretrained_cfg
#     if pretrained:
#         if not Path(model_path).is_file():
#             url = model.default_cfg['url']
#             torch.hub.download_url_to_file(url=url, dst=model_path)
#         model._load_state_dict(model_path)
#     return model
#
#
# @register_pip_model
# @register_model
# def mamba_vision_L2(pretrained=False, **kwargs):
#     model_path = kwargs.pop("model_path", "/tmp/mamba_vision_L2.pth.tar")
#     depths = kwargs.pop("depths", [3, 3, 12, 5])
#     num_heads = kwargs.pop("num_heads", [4, 8, 16, 32])
#     window_size = kwargs.pop("window_size", [8, 8, 14, 7])
#     dim = kwargs.pop("dim", 196)
#     in_dim = kwargs.pop("in_dim", 64)
#     mlp_ratio = kwargs.pop("mlp_ratio", 4)
#     resolution = kwargs.pop("resolution", 224)
#     drop_path_rate = kwargs.pop("drop_path_rate", 0.3)
#     layer_scale = kwargs.pop("layer_scale", 1e-5)
#     pretrained_cfg = resolve_pretrained_cfg('mamba_vision_L2').to_dict()
#     update_args(pretrained_cfg, kwargs, kwargs_filter=None)
#     model = MambaVision(depths=[3, 3, 12, 5],
#                         num_heads=[4, 8, 16, 32],
#                         window_size=[8, 8, 14, 7],
#                         dim=196,
#                         in_dim=64,
#                         mlp_ratio=4,
#                         resolution=224,
#                         drop_path_rate=0.3,
#                         layer_scale=1e-5,
#                         layer_scale_conv=None,
#                         **kwargs)
#     model.pretrained_cfg = pretrained_cfg
#     model.default_cfg = model.pretrained_cfg
#     if pretrained:
#         if not Path(model_path).is_file():
#             url = model.default_cfg['url']
#             torch.hub.download_url_to_file(url=url, dst=model_path)
#         model._load_state_dict(model_path)
#     return model


class SwinTransformerBlock(nn.Module):

    def __init__(self, dim, input_resolution, num_heads, window_size=7, shift_size=0,

                 mlp_ratio=4., qkv_bias=True, qk_scale=None, act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.num_heads = num_heads
        self.window_size = window_size
        self.shift_size = shift_size
        self.mlp_ratio = mlp_ratio
        if min(self.input_resolution) <= self.window_size:
            # if window size is larger than input resolution, we don't partition windows
            self.shift_size = 0
            self.window_size = min(self.input_resolution)
        assert 0 <= self.shift_size < self.window_size, "shift_size must in 0-window_size"

        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim, window_size=to_2tuple(self.window_size), num_heads=num_heads,
            qkv_bias=qkv_bias, qk_scale=qk_scale)

        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer)

        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        else:
            attn_mask = None

        self.register_buffer("attn_mask", attn_mask)

    def forward(self, x):

        H, W = self.input_resolution
        B, L, C = x.shape
        assert L == H * W, "input feature has wrong size"

        shortcut = x
        x = self.norm1(x)
        x = x.view(B, H, W, C)

        # cyclic shift
        if self.shift_size > 0:
            shifted_x = torch.roll(x, shifts=(-self.shift_size, -self.shift_size), dims=(1, 2))
        else:
            shifted_x = x

        # partition windows
        x_windows = window_partition(shifted_x, self.window_size)  # nW*B, window_size, window_size, C
        x_windows = x_windows.view(-1, self.window_size * self.window_size, C)  # nW*B, window_size*window_size, C
        B_, N, C = x_windows.shape

        # merge windows
        attn_windows = self.attn(x_windows,
                                 add_token=False,
                                 mask=self.attn_mask)
        attn_windows = attn_windows.view(-1, self.window_size, self.window_size, C)
        shifted_x = window_reverse(attn_windows, self.window_size, H, W)  # B H' W' C

        # reverse cyclic shift
        if self.shift_size > 0:
            x = torch.roll(shifted_x, shifts=(self.shift_size, self.shift_size), dims=(1, 2))
        else:
            x = shifted_x
        x = x.view(B, H * W, C)

        # FFN
        x = shortcut + x
        x = x + self.mlp(self.norm2(x))

        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, num_heads={self.num_heads}, " \
               f"window_size={self.window_size}, shift_size={self.shift_size}, mlp_ratio={self.mlp_ratio}"

    def flops(self):
        flops = 0
        H, W = self.input_resolution
        # norm1
        flops += self.dim * H * W
        # W-MSA/SW-MSA
        nW = H * W / self.window_size / self.window_size
        flops += nW * self.attn.flops(self.window_size * self.window_size)
        # mlp
        flops += 2 * H * W * self.dim * self.dim * self.mlp_ratio
        # norm2
        flops += self.dim * H * W
        return flops

    def update_mask(self):
        if self.shift_size > 0:
            # calculate attention mask for SW-MSA
            H, W = self.input_resolution
            img_mask = torch.zeros((1, H, W, 1))  # 1 H W 1
            h_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            w_slices = (slice(0, -self.window_size),
                        slice(-self.window_size, -self.shift_size),
                        slice(-self.shift_size, None))
            cnt = 0
            for h in h_slices:
                for w in w_slices:
                    img_mask[:, h, w, :] = cnt
                    cnt += 1

            mask_windows = window_partition(img_mask, self.window_size)  # nW, window_size, window_size, 1
            mask_windows = mask_windows.view(-1, self.window_size * self.window_size)
            attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
            attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
            self.attn_mask = attn_mask.cuda()
        else:
            pass

class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
                 downsample=None):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth
        self.blocks = nn.ModuleList([
            MambaVisionMixer(
                d_model=out_dim,
                d_state=num_heads,  # 假设 MambaVisionMixer 使用 num_heads 参数作为 d_state 参数
                d_conv=window_size,  # 适应 MambaVisionMixer 需要的卷积窗口大小
                expand=mlp_ratio,  # 使用 mlp_ratio 参数作为扩展因子
                window_size=window_size,
                input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
                layer_idx=i  # 假设 MambaVisionMixer 使用 layer_idx 参数
            )
            for i in range(depth)
        ])

        # patch merging layer
        if downsample is not None:
            self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.downsample = None

    def forward(self, x):
        if self.downsample is not None:
            x = self.downsample(x)
        for _, blk in enumerate(self.blocks):
            x = blk(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

    def flops(self):
        flops = 0
        for blk in self.blocks:
            flops += blk.flops()
        if self.downsample is not None:
            flops += self.downsample.flops()
        return flops

    def update_resolution(self, H, W):
        for _, blk in enumerate(self.blocks):
            blk.update_resolution(H // 2, W // 2)  # 假设需要适配 update_resolution 方法
        if self.downsample is not None:
            self.downsample.input_resolution = (H * 2, W * 2)



# class BasicLayer(nn.Module):
#     def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None, norm_layer=nn.LayerNorm,
#                  downsample=None):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(dim=out_dim,
#                                  input_resolution=(input_resolution[0] // 2, input_resolution[1] // 2),
#                                  num_heads=num_heads, window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                  norm_layer=norm_layer)
#             for i in range(depth)])
#
#         # patch merging layer
#         if downsample is not None:
#             self.downsample = downsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
#         else:
#             self.downsample = None
#
#     def forward(self, x):
#         if self.downsample is not None:
#             x = self.downsample(x)
#         for _, blk in enumerate(self.blocks):
#             x = blk(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
#
#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#         if self.downsample is not None:
#             flops += self.downsample.flops()
#         return flops
#
#     def update_resolution(self, H, W):
#         for _, blk in enumerate(self.blocks):
#             blk.input_resolution = (H, W)
#             blk.update_mask()
#         if self.downsample is not None:
#             self.downsample.input_resolution = (H * 2, W * 2)

class AdaptiveModulator(nn.Module):     #channle model自适应模块 的SM模块
    def __init__(self, M):
        super(AdaptiveModulator, self).__init__()
        self.fc = nn.Sequential(
            nn.Linear(1, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.ReLU(),
            nn.Linear(M, M),
            nn.Sigmoid()
        )

    def forward(self, snr):
        return self.fc(snr)

class WITT_Encoder(nn.Module):
    def __init__(self, img_size, patch_size, in_chans,
                 embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, patch_norm=True,
                 bottleneck_dim=16):
        super().__init__()
        self.num_layers = len(depths)
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.embed_dims = embed_dims
        self.in_chans = in_chans
        self.patch_size = patch_size
        self.patches_resolution = img_size
        self.H = img_size[0] // (2 ** self.num_layers)
        self.W = img_size[1] // (2 ** self.num_layers)

        # self.stem = CMTStem(
        #     kernel_size=3,
        #     in_channel=self.in_chans,
        #     out_channel=self.embed_dims[0],
        #     layers_num=2
        # )

        self.patch_embed = PatchEmbed(img_size, 2, 3, embed_dims[0])
        self.hidden_dim = int(self.embed_dims[len(embed_dims)-1] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()    #nn.ModuleList 则是用来管理多个 nn.Module 的容器，将多个 nn.Module 对象组织在一起，形成一个列表。
        self.sm_list = nn.ModuleList()    #
        self.sm_list.append(nn.Linear(self.embed_dims[len(embed_dims)-1], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[len(embed_dims)-1]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer - 1]) if i_layer != 0 else 3,
                               out_dim=int(embed_dims[i_layer]),
                               input_resolution=(self.patches_resolution[0] // (2 ** i_layer),
                                                 self.patches_resolution[1] // (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               downsample=PatchMerging if i_layer != 0 else None)
            print("Encoder ", layer.extra_repr())
            self.layers.append(layer)
        self.norm = norm_layer(embed_dims[-1])
        self.head_list = nn.Linear(embed_dims[-1], C)    #   c=16
        self.apply(self._init_weights)

    def forward(self, x, snr, model):
        B, C, H, W = x.size()
        device = x.get_device()

        # x = self.stem(x)

        x = self.patch_embed(x)
        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
        x = self.norm(x)

        if model == 'WITT':
            snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1) #第 0 维度上增加一个维度，使其从原先的 1 维向量变为 2 维矩阵。 .expand(B, -1) 操作则是将张量沿着指定的维度进行复制扩展，其中 B 是扩展后的维度大小，而 -1 表示该维度的大小保持不变
            for i in range(self.layer_num):
                if i == 0:  #头一个模块需要建立，后面是递归调用了
                    temp = self.sm_list[i](x.detach()) #创建一个新的 Tensor，该 Tensor 与原始的 Tensor x 共享数据存储空间，但是与计算图分离，不再参与梯度计算。
                else:
                    temp = self.sm_list[i](temp)   #对 temp 应用第 i 个神经网络模块，并将结果存储回 temp 中。

                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, H * W // (self.num_layers ** 4), -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val

        x = self.head_list(x)
        return x

    def _init_weights(self, m):   #网络模型初始化权重参数
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)   #全连接层
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):     #归一化层
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):    #计算浮点数运算次数
        flops = 0
        flops += self.patch_embed.flops()
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        flops += self.num_features * self.patches_resolution[0] * self.patches_resolution[1] // (2 ** self.num_layers)
        return flops

    def update_resolution(self, H, W):    #更新图像长宽
        self.input_resolution = (H, W)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H // (2 ** (i_layer + 1)),
                                    W // (2 ** (i_layer + 1)))

    #网络的实例化
def create_encoder(**kwargs):    #**kwargs 用来接收不定数量的关键字参数，并将它们存储在一个字典中。
    model = WITT_Encoder(**kwargs)
    return model


def build_model(config):
    input_image = torch.ones([1, 256, 256]).to(config.device)
    model = create_encoder(**config.encoder_kwargs)
    model(input_image)
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()   #.numel()方法用于返回该张量中元素的总数。这在神经网络的参数数量统计以及模型大小分析中非常有用。
    print("TOTAL Params {}M".format(num_params / 10 ** 6))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))






