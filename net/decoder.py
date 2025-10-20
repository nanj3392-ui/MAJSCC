from datetime import datetime
from net.modules import *
import torch
from net.encoder import SwinTransformerBlock, AdaptiveModulator

import numpy as np
import torch.nn as nn
import torch

from torch.nn.modules import module
import torch.nn.functional as F
from models.encoders.vmamba import CVSSDecoderBlock
import torch.utils.checkpoint as checkpoint
from einops import rearrange


class PatchExpand(nn.Module):
    def __init__(self, input_resolution, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.expand = nn.Linear(dim, 2 * dim, bias=False) if dim_scale == 2 else nn.Identity()
        self.norm = norm_layer(dim // dim_scale)

    def forward(self, x):
        """
        x: B, H, W, C
        """

        x = self.expand(x)  # B, H, W, 2C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=2, p2=2, c=C // 4)
        x = self.norm(x)

        return x


class UpsampleExpand(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear = nn.Linear(dim, dim // 2, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(dim // 2)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        x = self.linear(x).permute(0, 3, 1, 2).contiguous()  # B, C/2, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3,
                                                                                           1).contiguous()  # B, 2H, 2W, C/2
        x = self.norm(x)
        return x


class FinalPatchExpand_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.expand = nn.Linear(dim, patch_size * patch_size * dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        x = self.expand(x)  # B, H, W, 16C
        B, H, W, C = x.shape
        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.patch_size, p2=self.patch_size,
                      c=C // (self.patch_size ** 2))
        x = self.norm(x)
        return x


class FinalUpsample_X4(nn.Module):
    def __init__(self, input_resolution, dim, patch_size=4, norm_layer=nn.LayerNorm):
        super().__init__()
        self.input_resolution = input_resolution
        self.dim = dim
        self.patch_size = patch_size
        self.linear1 = nn.Linear(dim, dim, bias=False)
        self.linear2 = nn.Linear(dim, dim, bias=False)
        self.output_dim = dim
        self.norm = norm_layer(self.output_dim)

    def forward(self, x):
        """
        x: B, H, W, C
        """
        B, H, W, C = x.shape
        x = self.linear1(x).permute(0, 3, 1, 2).contiguous()  # B, C, H, W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3,
                                                                                           1).contiguous()  # B, 2H, 2W, C
        x = self.linear2(x).permute(0, 3, 1, 2).contiguous()  # B, C, 2H, 2W
        x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False).permute(0, 2, 3,
                                                                                           1).contiguous()  # B, 4H, 4W, C
        x = self.norm(x)
        return x


class Mamba_up(nn.Module):
    def __init__(self, dim, input_resolution, depth, dt_rank="auto",
                 d_state=4, ssm_ratio=2.0, attn_drop_rate=0.,
                 drop_rate=0.0, mlp_ratio=4.0,
                 drop_path=0.1, norm_layer=nn.LayerNorm, upsample=None,
                 shared_ssm=False, softmax_version=False,
                 use_checkpoint=False, **kwargs):

        super().__init__()
        self.input_resolution = input_resolution
        self.depth = depth
        self.use_checkpoint = use_checkpoint

        # build blocks
        self.blocks = nn.ModuleList([
            CVSSDecoderBlock(
                hidden_dim=dim,
                drop_path=drop_path[i],
                norm_layer=norm_layer,
                attn_drop_rate=attn_drop_rate,
                d_state=d_state,
                dt_rank=dt_rank,
                ssm_ratio=ssm_ratio,
                shared_ssm=shared_ssm,
                softmax_version=softmax_version,
                use_checkpoint=use_checkpoint,
                mlp_ratio=mlp_ratio,
                act_layer=nn.GELU,
                drop=drop_rate,
            )
            for i in range(depth)])

        # patch merging layer
        if upsample is not None:
            # self.upsample = PatchExpand(input_resolution, dim=dim, dim_scale=2, norm_layer=norm_layer)
            self.upsample = UpsampleExpand(input_resolution, dim=dim, patch_size=2, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        for blk in self.blocks:
            if self.use_checkpoint:
                x = checkpoint.checkpoint(blk, x)
            else:
                x = blk(x)
        if self.upsample is not None:
            x = self.upsample(x)
        return x


class MambaDecoder(nn.Module):
    def __init__(self,
                 img_size=[480, 640],
                 in_channels=[96, 192, 384, 768],  # [64, 128, 320, 512],
                 num_classes=40,
                 dropout_ratio=0.1,
                 embed_dim=96,
                 align_corners=False,
                 patch_size=4,
                 depths=[4, 4, 4, 4],
                 mlp_ratio=4.,
                 drop_rate=0.0,
                 attn_drop_rate=0.,
                 drop_path_rate=0.1,
                 norm_layer=nn.LayerNorm,
                 use_checkpoint=False,
                 deep_supervision=False,
                 **kwargs):
        super().__init__()

        self.num_classes = num_classes
        self.num_layers = len(depths)  # actually only three depths are used. The last feature is simply upexpanded
        self.mlp_ratio = mlp_ratio
        self.patch_size = patch_size
        self.patches_resolution = [img_size[0] // patch_size, img_size[1] // patch_size]
        self.deep_supervision = deep_supervision

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]

        self.layers_up = nn.ModuleList()
        for i_layer in range(self.num_layers):
            if i_layer == 0:
                # B, 768, 15, 20 -> B, 384, 30, 40
                layer_up = PatchExpand(
                    input_resolution=(self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                      self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                    dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                    dim_scale=2,
                    norm_layer=norm_layer)
            else:
                # B, 30, 40, 384 -> B, 60, 80, 192
                # B, 60, 80, 192 -> B, 120, 160, 96
                # B, 120, 160, 96 -> B, 120, 160, 96
                layer_up = Mamba_up(dim=int(embed_dim * 2 ** (self.num_layers - 1 - i_layer)),
                                    input_resolution=(
                                        self.patches_resolution[0] // (2 ** (self.num_layers - 1 - i_layer)),
                                        self.patches_resolution[1] // (2 ** (self.num_layers - 1 - i_layer))),
                                    depth=depths[(self.num_layers - 1 - i_layer)],
                                    mlp_ratio=self.mlp_ratio,
                                    drop=drop_rate,
                                    attn_drop=attn_drop_rate,
                                    drop_path=dpr[sum(depths[:(self.num_layers - 1 - i_layer)]):sum(
                                        depths[:(self.num_layers - 1 - i_layer) + 1])],
                                    norm_layer=norm_layer,
                                    upsample=PatchExpand if (i_layer < self.num_layers - 1) else None,
                                    use_checkpoint=use_checkpoint)
            self.layers_up.append(layer_up)

        self.norm_up = norm_layer(embed_dim)
        if self.deep_supervision:
            self.norm_ds = nn.ModuleList([norm_layer(embed_dim * 2 ** (self.num_layers - 2 - i_layer)) for i_layer in
                                          range(self.num_layers - 1)])
            self.output_ds = nn.ModuleList([nn.Conv2d(in_channels=embed_dim * 2 ** (self.num_layers - 2 - i_layer),
                                                      out_channels=self.num_classes, kernel_size=1, bias=False) for
                                            i_layer in range(self.num_layers - 1)])

        # print("---final upsample expand_first---")
        # self.up = FinalPatchExpand_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
        #                                 patch_size=4, dim=embed_dim)
        self.up = FinalUpsample_X4(input_resolution=(img_size[0] // patch_size, img_size[1] // patch_size),
                                   patch_size=4, dim=embed_dim)
        self.output = nn.Conv2d(in_channels=embed_dim, out_channels=self.num_classes, kernel_size=1, bias=False)

    def forward_up_features(self, inputs):  # B, C, H, W
        if not self.deep_supervision:
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x = inputs[3 - inx]  # B, 768, 15, 20
                    x = x.permute(0, 2, 3, 1).contiguous()  # B, 15, 20, 768
                    y = layer_up(x)  # B, 30, 40, 384
                else:
                    # interpolate y to input size (only pst900 dataset needs)
                    B, C, H, W = inputs[3 - inx].shape
                    y = F.interpolate(y.permute(0, 3, 1, 2).contiguous(), size=(H, W), mode='bilinear',
                                      align_corners=False).permute(0, 2, 3, 1).contiguous()

                    x = y + inputs[3 - inx].permute(0, 2, 3, 1).contiguous()
                    y = layer_up(x)

            x = self.norm_up(y)

            return x
        else:
            # if deep supervision
            x_upsample = []
            for inx, layer_up in enumerate(self.layers_up):
                if inx == 0:
                    x = inputs[3 - inx]  # B, 768, 15, 20
                    x = x.permute(0, 2, 3, 1).contiguous()  # B, 15, 20, 768
                    y = layer_up(x)  # B, 30, 40, 384
                    x_upsample.append(self.norm_ds[inx](y))
                else:
                    x = y + inputs[3 - inx].permute(0, 2, 3, 1).contiguous()
                    y = layer_up(x)
                    if inx != self.num_layers - 1:
                        x_upsample.append((self.norm_ds[inx](y)))

            x = self.norm_up(y)

            return x, x_upsample

    def forward(self, inputs):
        if not self.deep_supervision:
            x = self.forward_up_features(inputs)  # B, H, W, C
            x_last = self.up_x4(x, self.patch_size)
            return x_last
        else:
            x, x_upsample = self.forward_up_features(inputs)
            x_last = self.up_x4(x, self.patch_size)
            x_output_0 = self.output_ds[0](
                F.interpolate(x_upsample[0].permute(0, 3, 1, 2).contiguous(), scale_factor=16, mode='bilinear',
                              align_corners=False))
            x_output_1 = self.output_ds[1](
                F.interpolate(x_upsample[1].permute(0, 3, 1, 2).contiguous(), scale_factor=8, mode='bilinear',
                              align_corners=False))
            x_output_2 = self.output_ds[2](
                F.interpolate(x_upsample[2].permute(0, 3, 1, 2).contiguous(), scale_factor=4, mode='bilinear',
                              align_corners=False))
            return x_last, x_output_0, x_output_1, x_output_2

    def up_x4(self, x, pz):
        B, H, W, C = x.shape

        x = self.up(x)
        x = x.view(B, pz * H, pz * W, -1)
        x = x.permute(0, 3, 1, 2).contiguous()  # B, C, 4H, 4W
        x = self.output(x)

        return x

class BasicLayer(nn.Module):
    def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
                 mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, upsample=None,):

        super().__init__()
        self.dim = dim
        self.input_resolution = input_resolution
        self.depth = depth

        # 使用 MambaDecoder 替代 SwinTransformerBlock
        self.blocks = MambaDecoder(
            img_size=input_resolution,  # 假设 input_resolution 是元组或列表
            in_channels=[dim for _ in range(depth)],  # 需要适配这个参数
            num_classes=out_dim,  # 假设输出维度对应分类任务的类数
            depths=[depth // 4] * 4,  # 将 depth 平均分配给各层
            num_heads=[num_heads] * depth,  # 分配 heads
            patch_size=window_size,  # 使用 window_size 作为 patch 大小
            mlp_ratio=mlp_ratio,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            norm_layer=norm_layer
        )

        # patch merging layer
        if upsample is not None:
            self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
        else:
            self.upsample = None

    def forward(self, x):
        x = self.blocks(x)  # 假设 MambaDecoder 的 forward 方法接受并返回相同形式的 x

        if self.upsample is not None:
            x = self.upsample(x)
        return x

    def extra_repr(self) -> str:
        return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"

# class BasicLayer(nn.Module):
#
#     def __init__(self, dim, out_dim, input_resolution, depth, num_heads, window_size,
#                  mlp_ratio=4., qkv_bias=True, qk_scale=None,
#                  norm_layer=nn.LayerNorm, upsample=None,):
#
#         super().__init__()
#         self.dim = dim
#         self.input_resolution = input_resolution
#         self.depth = depth
#
#         # build blocks
#         self.blocks = nn.ModuleList([
#             SwinTransformerBlock(dim=dim, input_resolution=input_resolution,
#                                  num_heads=num_heads, window_size=window_size,
#                                  shift_size=0 if (i % 2 == 0) else window_size // 2,
#                                  mlp_ratio=mlp_ratio,
#                                  qkv_bias=qkv_bias, qk_scale=qk_scale,
#                                  norm_layer=norm_layer)
#             for i in range(depth)])
#
#         # patch merging layer
#         if upsample is not None:
#             self.upsample = upsample(input_resolution, dim=dim, out_dim=out_dim, norm_layer=norm_layer)
#         else:
#             self.upsample = None
#
#     def forward(self, x):
#         for _, blk in enumerate(self.blocks):
#             x = blk(x)
#
#         if self.upsample is not None:
#             x = self.upsample(x)
#         return x
#
#     def extra_repr(self) -> str:
#         return f"dim={self.dim}, input_resolution={self.input_resolution}, depth={self.depth}"
#
#     def flops(self):
#         flops = 0
#         for blk in self.blocks:
#             flops += blk.flops()
#             print("blk.flops()", blk.flops())
#         if self.upsample is not None:
#             flops += self.upsample.flops()
#             print("upsample.flops()", self.upsample.flops())
#         return flops
#
#     def update_resolution(self, H, W):
#         self.input_resolution = (H, W)
#         for _, blk in enumerate(self.blocks):
#             blk.input_resolution = (H, W)
#             blk.update_mask()
#         if self.upsample is not None:
#             self.upsample.input_resolution = (H, W)


class WITT_Decoder(nn.Module):
    def __init__(self, img_size, embed_dims, depths, num_heads, C,
                 window_size=4, mlp_ratio=4., qkv_bias=True, qk_scale=None,
                 norm_layer=nn.LayerNorm, ape=False, patch_norm=True,
                 bottleneck_dim=16):
        super().__init__()

        self.num_layers = len(depths)
        self.ape = ape
        self.embed_dims = embed_dims
        self.patch_norm = patch_norm
        self.num_features = bottleneck_dim
        self.mlp_ratio = mlp_ratio
        self.H = img_size[0]
        self.W = img_size[1]
        self.patches_resolution = (img_size[0] // 2 ** len(depths), img_size[1] // 2 ** len(depths))
        num_patches = self.H // 4 * self.W // 4
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[0]))
            trunc_normal_(self.absolute_pos_embed, std=.02)

        # build layers
        self.layers = nn.ModuleList()
        for i_layer in range(self.num_layers):
            layer = BasicLayer(dim=int(embed_dims[i_layer]),
                               out_dim=int(embed_dims[i_layer + 1]) if (i_layer < self.num_layers - 1) else 3,
                               input_resolution=(self.patches_resolution[0] * (2 ** i_layer),
                                                 self.patches_resolution[1] * (2 ** i_layer)),
                               depth=depths[i_layer],
                               num_heads=num_heads[i_layer],
                               window_size=window_size,
                               mlp_ratio=self.mlp_ratio,
                               qkv_bias=qkv_bias, qk_scale=qk_scale,
                               norm_layer=norm_layer,
                               upsample=PatchReverseMerging)
            self.layers.append(layer)
            print("Decoder ", layer.extra_repr())
        self.head_list = nn.Linear(C, embed_dims[0])
        self.apply(self._init_weights)

        self.hidden_dim = int(self.embed_dims[0] * 1.5)
        self.layer_num = layer_num = 7
        self.bm_list = nn.ModuleList()
        self.sm_list = nn.ModuleList()
        self.sm_list.append(nn.Linear(self.embed_dims[0], self.hidden_dim))
        for i in range(layer_num):
            if i == layer_num - 1:
                outdim = self.embed_dims[0]
            else:
                outdim = self.hidden_dim
            self.bm_list.append(AdaptiveModulator(self.hidden_dim))
            self.sm_list.append(nn.Linear(self.hidden_dim, outdim))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, snr, model):
        B, L, C = x.size()
        device = x.get_device()
        x = self.head_list(x)

        if model == 'WITT':
            # token modulation according to input snr value
            snr_cuda = torch.tensor(snr, dtype=torch.float).to(device)
            snr_batch = snr_cuda.unsqueeze(0).expand(B, -1)
            for i in range(self.layer_num):
                if i == 0:
                    temp = self.sm_list[i](x.detach())
                else:
                    temp = self.sm_list[i](temp)
                bm = self.bm_list[i](snr_batch).unsqueeze(1).expand(-1, L, -1)
                temp = temp * bm
            mod_val = self.sigmoid(self.sm_list[-1](temp))
            x = x * mod_val

        for i_layer, layer in enumerate(self.layers):
            x = layer(x)
        B, L, N = x.shape
        x = x.reshape(B, self.H, self.W, N).permute(0, 3, 1, 2)

        return x

    def _init_weights(self, m):     #权重初始化
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'absolute_pos_embed'}

    @torch.jit.ignore
    def no_weight_decay_keywords(self):
        return {'relative_position_bias_table'}

    def flops(self):
        flops = 0
        for i, layer in enumerate(self.layers):
            flops += layer.flops()
        return flops

    def update_resolution(self, H, W):
        self.input_resolution = (H, W)
        self.H = H * 2 ** len(self.layers)
        self.W = W * 2 ** len(self.layers)
        for i_layer, layer in enumerate(self.layers):
            layer.update_resolution(H * (2 ** i_layer),
                                    W * (2 ** i_layer))


def create_decoder(**kwargs):
    model = WITT_Decoder(**kwargs)
    return model


def build_model(config):
    input_image = torch.ones([1, 1536, 256]).to(config.device)
    model = create_decoder(**config.encoder_kwargs).to(config.device)
    t0 = datetime.datetime.now()
    with torch.no_grad():
        for i in range(100):
            features = model(input_image, SNR=15)
        t1 = datetime.datetime.now()
        delta_t = t1 - t0
        print("Decoding Time per img {}s".format((delta_t.seconds + 1e-6 * delta_t.microseconds) / 100))
    print("TOTAL FLOPs {}G".format(model.flops() / 10 ** 9))
    num_params = 0
    for param in model.parameters():
        num_params += param.numel()
    print("TOTAL Params {}M".format(num_params / 10 ** 6))

