import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
# from timm.models.registry import register_model
# from timm.models.vision_transformer import _cfg
# from mmseg.models.builder import BACKBONES
# from mmseg.utils import get_root_logger
from logging import getLogger as get_root_logger
from mmengine.runner import load_checkpoint
from models.resnet import BasicBlock
from models.FCTransformer import CTransformer
from models.cc_attention import CrossCrissCrossAttention
from models.DualFusion import FusionModelV2
from models.BitemporalFusion import BitFusion

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.sr_ratio = sr_ratio
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        B, N, C = x.shape
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x


class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))

        return x


class OverlapPatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=224, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride,
                              padding=(patch_size[0] // 2, patch_size[1] // 2))
        self.norm = nn.LayerNorm(embed_dim)

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        x = self.proj(x)
        _, _, H, W = x.shape
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)

        return x, H, W


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class MixVisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths


        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])



        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for i in range(self.depths[0]):
            self.block1[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[0]
        for i in range(self.depths[1]):
            self.block2[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[1]
        for i in range(self.depths[2]):
            self.block3[i].drop_path.drop_prob = dpr[cur + i]

        cur += self.depths[2]
        for i in range(self.depths[3]):
            self.block4[i].drop_path.drop_prob = dpr[cur + i]

    def freeze_patch_emb(self):
        self.patch_embed1.requires_grad = False

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)


        return x


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W)
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x


class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out * x


class HybridEncoderV2(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 res_norm_layer=None, strides=None, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])



        # resnet
        if res_norm_layer is None:
            res_norm_layer = nn.BatchNorm2d
        self._norm_layer = res_norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=self.strides[0], padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.res_layer1 = self._make_layer(block, 64, layers[0])
        self.res_layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                           dilate=replace_stride_with_dilation[0])
        self.res_layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[3],
                                           dilate=replace_stride_with_dilation[1])
        self.res_layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[4],
                                           dilate=replace_stride_with_dilation[2])
        # self.converge_layer1 = nn.Sequential(ChannelAttention(embed_dims[0] * 2),
        #                                      nn.Conv2d(embed_dims[0] * 2, embed_dims[0], kernel_size=3, padding=1,
        #                                                stride=1), nn.BatchNorm2d(embed_dims[0]), nn.ReLU())
        # self.converge_layer2 = nn.Sequential(ChannelAttention(embed_dims[1] * 2),
        #                                      nn.Conv2d(embed_dims[1] * 2, embed_dims[1], kernel_size=3, padding=1,
        #                                                stride=1), nn.BatchNorm2d(embed_dims[1]), nn.ReLU())
        # self.converge_layer3 = nn.Sequential(ChannelAttention(embed_dims[2] * 2),
        #                                      nn.Conv2d(embed_dims[2] * 2, embed_dims[2], kernel_size=3, padding=1,
        #                                                stride=1), nn.BatchNorm2d(embed_dims[2]), nn.ReLU())
        # self.converge_layer4 = nn.Sequential(ChannelAttention(embed_dims[3] * 2),
        #                                      nn.Conv2d(embed_dims[3] * 2, embed_dims[3], kernel_size=3, padding=1,
        #                                                stride=1), nn.BatchNorm2d(embed_dims[3]), nn.ReLU())
        self.converge_layer_list = nn.ModuleList([nn.Sequential(nn.Conv2d(embed_dims[i] * 2, embed_dims[i],
                                                                          kernel_size=3, padding=1, stride=1),
                                                                nn.BatchNorm2d(embed_dims[i]), nn.ReLU()) for i in
                                                  range(len(embed_dims))])

        self.cross_att_list = nn.ModuleList(
            [CrossCrissCrossAttention(embed_dims[i]) for i in range(len(embed_dims))])
        # resnet init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # segformer init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # resnet method
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _trans_forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def _res_forward_features(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layer1(x)

        outs.append(x)
        x = self.res_layer2(x)
        outs.append(x)
        x = self.res_layer3(x)
        outs.append(x)
        x = self.res_layer4(x)
        outs.append(x)
        return outs

    def _conver_features(self, res_features, trans_features):
        outs = []

        for i, converge_layer in enumerate(self.converge_layer_list):
            out = converge_layer(self.cross_att_list[i](res_features[i], trans_features[i]))
            outs.append(out)

        return outs

    def forward(self, x1, x2):
        trans_outs1 = self._trans_forward_features(x1)
        trans_outs2 = self._trans_forward_features(x2)
        res_outs1 = self._res_forward_features(x1)
        res_outs2 = self._res_forward_features(x2)
        conver_outs1 = self._conver_features(res_outs1, trans_outs1)
        conver_outs2 = self._conver_features(res_outs2, trans_outs2)
        return conver_outs1, conver_outs2


class HybridEncoder(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 res_norm_layer=None, strides=None, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])



        # resnet
        if res_norm_layer is None:
            res_norm_layer = nn.BatchNorm2d
        self._norm_layer = res_norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=self.strides[0], padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.res_layer1 = self._make_layer(block, 64, layers[0])
        self.res_layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                           dilate=replace_stride_with_dilation[0])
        self.res_layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[3],
                                           dilate=replace_stride_with_dilation[1])
        self.res_layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[4],
                                           dilate=replace_stride_with_dilation[2])
        # self.converge_layer1 = nn.Sequential(ChannelAttention(embed_dims[0] * 2),
        #                                      nn.Conv2d(embed_dims[0] * 2, embed_dims[0], kernel_size=3, padding=1,
        #                                                stride=1), nn.BatchNorm2d(embed_dims[0]), nn.ReLU())
        # self.converge_layer2 = nn.Sequential(ChannelAttention(embed_dims[1] * 2),
        #                                      nn.Conv2d(embed_dims[1] * 2, embed_dims[1], kernel_size=3, padding=1,
        #                                                stride=1), nn.BatchNorm2d(embed_dims[1]), nn.ReLU())
        # self.converge_layer3 = nn.Sequential(ChannelAttention(embed_dims[2] * 2),
        #                                      nn.Conv2d(embed_dims[2] * 2, embed_dims[2], kernel_size=3, padding=1,
        #                                                stride=1), nn.BatchNorm2d(embed_dims[2]), nn.ReLU())
        # self.converge_layer4 = nn.Sequential(ChannelAttention(embed_dims[3] * 2),
        #                                      nn.Conv2d(embed_dims[3] * 2, embed_dims[3], kernel_size=3, padding=1,
        #                                                stride=1), nn.BatchNorm2d(embed_dims[3]), nn.ReLU())
        # self.converge_layer_list = nn.ModuleList([nn.Sequential(ChannelAttention(embed_dims[i] * 2),
        #                                                         nn.Conv2d(embed_dims[i] * 2, embed_dims[i],
        #                                                                   kernel_size=3, padding=1, stride=1),
        #                                                         nn.BatchNorm2d(embed_dims[i]), nn.ReLU()) for i in
        #                                           range(len(embed_dims))])
        self.converge_layer_list = nn.ModuleList([nn.Sequential(nn.Conv2d(embed_dims[i] * 2, embed_dims[i],kernel_size=3, padding=1, stride=1),nn.BatchNorm2d(embed_dims[i]), nn.ReLU()) for i in range(len(embed_dims))])
        # resnet init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # segformer init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # resnet method
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _trans_forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def _res_forward_features(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layer1(x)

        outs.append(x)
        x = self.res_layer2(x)
        outs.append(x)
        x = self.res_layer3(x)
        outs.append(x)
        x = self.res_layer4(x)
        outs.append(x)
        return outs

    def _conver_features(self, res_features, trans_features):
        outs = []

        for i, converge_layer in enumerate(self.converge_layer_list):
            out = converge_layer(torch.cat([res_features[i], trans_features[i]], dim=1))
            outs.append(out)

        return outs

    def forward(self, x1, x2):
        trans_outs1 = self._trans_forward_features(x1)
        trans_outs2 = self._trans_forward_features(x2)
        res_outs1 = self._res_forward_features(x1)
        res_outs2 = self._res_forward_features(x2)
        conver_outs1 = self._conver_features(res_outs1, trans_outs1)
        conver_outs2 = self._conver_features(res_outs2, trans_outs2)
        return conver_outs1, conver_outs2

class HybridEncoder_SCM(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 res_norm_layer=None, strides=None, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])



        # resnet
        if res_norm_layer is None:
            res_norm_layer = nn.BatchNorm2d
        self._norm_layer = res_norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:

            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=self.strides[0], padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.res_layer1 = self._make_layer(block, 64, layers[0])
        self.res_layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                           dilate=replace_stride_with_dilation[0])
        self.res_layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[3],
                                           dilate=replace_stride_with_dilation[1])
        self.res_layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[4],
                                           dilate=replace_stride_with_dilation[2])
        self.converge_layer_list = nn.ModuleList([nn.Sequential(
            nn.Conv2d(embed_dims[i] * 2, embed_dims[i], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(embed_dims[i]), nn.ReLU()) for i in range(len(embed_dims))])
        #######################################################EncoderV3G7################################################
        self.cross_att_list = nn.ModuleList(
            [CrossCrissCrossAttention(embed_dims[i]) for i in range(len(embed_dims))])
        #################################################################################################################
        # resnet init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # segformer init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # resnet method
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _trans_forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def _res_forward_features(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layer1(x)

        outs.append(x)
        x = self.res_layer2(x)
        outs.append(x)
        x = self.res_layer3(x)
        outs.append(x)
        x = self.res_layer4(x)
        outs.append(x)
        return outs

    def _fusion_forward_features(self, x):
        # stage1 resnet
        res_outs = []
        res_x = x
        res_x = self.conv1(res_x)
        res_x = self.bn1(res_x)
        res_x = self.relu(res_x)
        res_x = self.maxpool(res_x)
        res_x = self.res_layer1(res_x)
        # stage 1 transformer
        B = x.shape[0]
        trans_outs = []
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        res_x, x = self.cross_att_list[0](res_x, x)
        trans_outs.append(x)
        res_outs.append(res_x)
        # stage2 resnet
        res_x = self.res_layer2(res_x)
        # stage 2 transformer
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res_x, x = self.cross_att_list[1](res_x, x)
        res_outs.append(res_x)
        trans_outs.append(x)
        # stage3 resnet
        res_x = self.res_layer3(res_x)
        # stage 3 transformer
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res_x, x = self.cross_att_list[2](res_x, x)
        trans_outs.append(x)
        res_outs.append(res_x)
        # stage4 resnet
        res_x = self.res_layer4(res_x)

        # stage 4 transformer
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res_x, x = self.cross_att_list[3](res_x, x)
        trans_outs.append(x)
        res_outs.append(res_x)
        return res_outs, trans_outs

    def _conver_features(self, res_features, trans_features):
        outs = []

        for i, converge_layer in enumerate(self.converge_layer_list):
            out = converge_layer(torch.cat([res_features[i], trans_features[i]], dim=1))
            outs.append(out)

        return outs

    def forward(self, x1, x2):

        res_outs1, trans_outs1 = self._fusion_forward_features(x1)
        res_outs2, trans_outs2 = self._fusion_forward_features(x2)
        conver_outs1 = self._conver_features(res_outs1, trans_outs1)
        conver_outs2 = self._conver_features(res_outs2, trans_outs2)
        return conver_outs1, conver_outs2



class HybridEncoderV3(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 res_norm_layer=None, strides=None, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        # classification head
        # self.head = nn.Linear(embed_dims[3], num_classes) if num_classes > 0 else nn.Identity()

        # resnet
        if res_norm_layer is None:
            res_norm_layer = nn.BatchNorm2d
        self._norm_layer = res_norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=self.strides[0], padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.res_layer1 = self._make_layer(block, 64, layers[0])
        self.res_layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                           dilate=replace_stride_with_dilation[0])
        self.res_layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[3],
                                           dilate=replace_stride_with_dilation[1])
        self.res_layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[4],
                                           dilate=replace_stride_with_dilation[2])
        ################################################EncoderV3G8###################################################
        # self.converge_layer_list = nn.ModuleList([FusionModel(embed_dims[i]) for i in
        #                                           range(len(embed_dims))])
        ###############################################################################################################
        ################################################EncoderV3G9###################################################
        self.converge_layer_list = nn.ModuleList([FusionModelV2(embed_dims[i]) for i in
                                                  range(len(embed_dims))])
        ###############################################################################################################
        # self.cross_att_list = nn.ModuleList(
        #     [_CrNonLocalBlock(in_channels=embed_dims[i], inter_channels=embed_dims[i]) for i in range(len(embed_dims))])
        #######################################################EncoderV3G4################################################
        # self.cross_att_list = nn.ModuleList(
        #     [CTransformer(out_channels=embed_dims[i],num_heads=2) for i in range(len(embed_dims))])
        #################################################################################################################
        #######################################################EncoderV3G7################################################
        self.cross_att_list = nn.ModuleList(
            [CrossCrissCrossAttention(embed_dims[i]) for i in range(len(embed_dims))])
        #################################################################################################################
        # resnet init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # segformer init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # resnet method
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _trans_forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def _res_forward_features(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layer1(x)

        outs.append(x)
        x = self.res_layer2(x)
        outs.append(x)
        x = self.res_layer3(x)
        outs.append(x)
        x = self.res_layer4(x)
        outs.append(x)
        return outs

    def _fusion_forward_features(self, x):
        # stage1 resnet
        res_outs = []
        res_x = x
        res_x = self.conv1(res_x)
        res_x = self.bn1(res_x)
        res_x = self.relu(res_x)
        res_x = self.maxpool(res_x)
        res_x = self.res_layer1(res_x)
        # stage 1 transformer
        B = x.shape[0]
        trans_outs = []
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        res_x, x = self.cross_att_list[0](res_x, x)
        trans_outs.append(x)
        res_outs.append(res_x)
        # stage2 resnet
        res_x = self.res_layer2(res_x)
        # stage 2 transformer
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res_x, x = self.cross_att_list[1](res_x, x)
        res_outs.append(res_x)
        trans_outs.append(x)
        # stage3 resnet
        res_x = self.res_layer3(res_x)
        # stage 3 transformer
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res_x, x = self.cross_att_list[2](res_x, x)
        trans_outs.append(x)
        res_outs.append(res_x)
        # stage4 resnet
        res_x = self.res_layer4(res_x)

        # stage 4 transformer
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        res_x, x = self.cross_att_list[3](res_x, x)
        trans_outs.append(x)
        res_outs.append(res_x)
        return res_outs, trans_outs

    def _conver_features(self, res_features, trans_features):
        outs = []

        #############################EncoderV3G8##########################
        for i, converge_layer in enumerate(self.converge_layer_list):
            out = converge_layer(res_features[i], trans_features[i])
            outs.append(out)
        ##################################################################
        return outs

    def forward(self, x1, x2):

        res_outs1, trans_outs1 = self._fusion_forward_features(x1)
        res_outs2, trans_outs2 = self._fusion_forward_features(x2)
        conver_outs1 = self._conver_features(res_outs1, trans_outs1)
        conver_outs2 = self._conver_features(res_outs2, trans_outs2)
        return conver_outs1, conver_outs2


class HybridEncoderV3_SingalTrans(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 res_norm_layer=None, strides=None, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm,
                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths

        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(img_size=img_size, patch_size=7, stride=4, in_chans=in_chans,
                                              embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(img_size=img_size // 4, patch_size=3, stride=2, in_chans=embed_dims[0],
                                              embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(img_size=img_size // 8, patch_size=3, stride=2, in_chans=embed_dims[1],
                                              embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(img_size=img_size // 16, patch_size=3, stride=2, in_chans=embed_dims[2],
                                              embed_dim=embed_dims[3])

        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(
            dim=embed_dims[0], num_heads=num_heads[0], mlp_ratio=mlp_ratios[0], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[0])
            for i in range(depths[0])])
        self.norm1 = norm_layer(embed_dims[0])

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(
            dim=embed_dims[1], num_heads=num_heads[1], mlp_ratio=mlp_ratios[1], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[1])
            for i in range(depths[1])])
        self.norm2 = norm_layer(embed_dims[1])

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(
            dim=embed_dims[2], num_heads=num_heads[2], mlp_ratio=mlp_ratios[2], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[2])
            for i in range(depths[2])])
        self.norm3 = norm_layer(embed_dims[2])

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(
            dim=embed_dims[3], num_heads=num_heads[3], mlp_ratio=mlp_ratios[3], qkv_bias=qkv_bias, qk_scale=qk_scale,
            drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + i], norm_layer=norm_layer,
            sr_ratio=sr_ratios[3])
            for i in range(depths[3])])
        self.norm4 = norm_layer(embed_dims[3])

        ################################################EncoderV3G8###################################################
        # self.converge_layer_list = nn.ModuleList([FusionModel(embed_dims[i]) for i in
        #                                           range(len(embed_dims))])
        ###############################################################################################################
        ################################################EncoderV3G9###################################################
        self.converge_layer_list = nn.ModuleList([FusionModelV2(embed_dims[i]) for i in
                                                  range(len(embed_dims))])
        ###############################################################################################################
        # self.cross_att_list = nn.ModuleList(
        #     [_CrNonLocalBlock(in_channels=embed_dims[i], inter_channels=embed_dims[i]) for i in range(len(embed_dims))])
        #######################################################EncoderV3G4################################################
        # self.cross_att_list = nn.ModuleList(
        #     [CTransformer(out_channels=embed_dims[i],num_heads=2) for i in range(len(embed_dims))])
        #################################################################################################################
        #######################################################EncoderV3G7################################################
        self.cross_att_list = nn.ModuleList(
            [CrossCrissCrossAttention(embed_dims[i]) for i in range(len(embed_dims))])
        #################################################################################################################

        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # resnet method
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _trans_forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def _res_forward_features(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layer1(x)

        outs.append(x)
        x = self.res_layer2(x)
        outs.append(x)
        x = self.res_layer3(x)
        outs.append(x)
        x = self.res_layer4(x)
        outs.append(x)
        return outs

    def _fusion_forward_features(self, x):

        B = x.shape[0]
        trans_outs = []
        res_outs = []
        x_Y = x
        # stage 1 transformer
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        # TRANS_y_1
        x_Y, H_Y, W_Y = self.patch_embed1_Y(x_Y)
        for i, blk in enumerate(self.block1_Y):
            x_Y = blk(x_Y, H_Y, W_Y)
        x_Y = self.norm1_Y(x_Y)
        res_x = x_Y.reshape(B, H_Y, W_Y, -1).permute(0, 3, 1, 2).contiguous()

        res_x, x = self.cross_att_list[0](res_x, x)
        trans_outs.append(x)
        res_outs.append(res_x)

        # stage 2 transformer
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage2 resnet
        # TRANS_y_2
        x_Y, H_Y, W_Y = self.patch_embed2_Y(res_x)
        for i, blk in enumerate(self.block2_Y):
            x_Y = blk(x_Y, H_Y, W_Y)
        x_Y = self.norm2_Y(x_Y)
        res_x = x_Y.reshape(B, H_Y, W_Y, -1).permute(0, 3, 1, 2).contiguous()

        res_x, x = self.cross_att_list[1](res_x, x)
        res_outs.append(res_x)
        trans_outs.append(x)



        # stage 3 transformer
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage3 resnet
        # TRANS_y_3
        x_Y, H_Y, W_Y = self.patch_embed3_Y(res_x)
        for i, blk in enumerate(self.block3_Y):
            x_Y = blk(x_Y, H_Y, W_Y)
        x_Y = self.norm3_Y(x_Y)
        res_x = x_Y.reshape(B, H_Y, W_Y, -1).permute(0, 3, 1, 2).contiguous()



        res_x, x = self.cross_att_list[2](res_x, x)
        trans_outs.append(x)
        res_outs.append(res_x)


        # stage 4 transformer
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage4 resnet
        # TRANS_y_4
        x_Y, H_Y, W_Y = self.patch_embed4_Y(res_x)
        for i, blk in enumerate(self.block4_Y):
            x_Y = blk(x_Y, H_Y, W_Y)
        x_Y = self.norm4_Y(x_Y)
        res_x = x_Y.reshape(B, H_Y, W_Y, -1).permute(0, 3, 1, 2).contiguous()



        res_x, x = self.cross_att_list[3](res_x, x)
        trans_outs.append(x)
        res_outs.append(res_x)
        return res_outs, trans_outs

    def _conver_features(self, res_features, trans_features):
        outs = []
        #############################EncoderV3G8##########################
        for i, converge_layer in enumerate(self.converge_layer_list):
            out = converge_layer(res_features[i], trans_features[i])
            outs.append(out)
        ##################################################################
        return outs

    def forward(self, x1, x2):
        trans_outs1 = self._trans_forward_features(x1)
        trans_outs2 = self._trans_forward_features(x2)

        return trans_outs1, trans_outs2



class HybridEncoderV3_SingalCNN(nn.Module):
    def __init__(self, block, layers, zero_init_residual=False, groups=1, width_per_group=64,
                 replace_stride_with_dilation=None,
                 res_norm_layer=None, strides=None, img_size=224, patch_size=16, in_chans=3, num_classes=1000,
                 embed_dims=[64, 128, 256, 512],

                 depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.inplanes = 64
        self.dilation = 1
        self.groups = groups
        self.base_width = width_per_group
        # resnet
        if res_norm_layer is None:
            res_norm_layer = nn.BatchNorm2d
        self._norm_layer = res_norm_layer

        self.strides = strides
        if self.strides is None:
            self.strides = [2, 2, 2, 2, 2]


        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))

        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=self.strides[0], padding=3,
                               bias=False)
        self.bn1 = self._norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=self.strides[1], padding=1)
        self.res_layer1 = self._make_layer(block, 64, layers[0])
        self.res_layer2 = self._make_layer(block, 128, layers[1], stride=self.strides[2],
                                           dilate=replace_stride_with_dilation[0])
        self.res_layer3 = self._make_layer(block, 256, layers[2], stride=self.strides[3],
                                           dilate=replace_stride_with_dilation[1])
        self.res_layer4 = self._make_layer(block, 512, layers[3], stride=self.strides[4],
                                           dilate=replace_stride_with_dilation[2])

        ################################################EncoderV3G8###################################################
        # self.converge_layer_list = nn.ModuleList([FusionModel(embed_dims[i]) for i in
        #                                           range(len(embed_dims))])
        ###############################################################################################################
        ################################################EncoderV3G9###################################################
        self.converge_layer_list = nn.ModuleList([FusionModelV2(embed_dims[i]) for i in
                                                  range(len(embed_dims))])
        ###############################################################################################################
        # self.cross_att_list = nn.ModuleList(
        #     [_CrNonLocalBlock(in_channels=embed_dims[i], inter_channels=embed_dims[i]) for i in range(len(embed_dims))])
        #######################################################EncoderV3G4################################################
        # self.cross_att_list = nn.ModuleList(
        #     [CTransformer(out_channels=embed_dims[i],num_heads=2) for i in range(len(embed_dims))])
        #################################################################################################################
        #######################################################EncoderV3G7################################################
        self.cross_att_list = nn.ModuleList(
            [CrossCrissCrossAttention(embed_dims[i]) for i in range(len(embed_dims))])
        #################################################################################################################
        # resnet init
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
        # segformer init
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    # resnet method
    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

    def _trans_forward_features(self, x):
        B = x.shape[0]
        outs = []

        # stage 1
        x, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            x = blk(x, H, W)
        x = self.norm1(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 2
        x, H, W = self.patch_embed2(x)
        for i, blk in enumerate(self.block2):
            x = blk(x, H, W)
        x = self.norm2(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 3
        x, H, W = self.patch_embed3(x)
        for i, blk in enumerate(self.block3):
            x = blk(x, H, W)
        x = self.norm3(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        # stage 4
        x, H, W = self.patch_embed4(x)
        for i, blk in enumerate(self.block4):
            x = blk(x, H, W)
        x = self.norm4(x)
        x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        outs.append(x)

        return outs

    def _res_forward_features(self, x):
        outs = []
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.res_layer1(x)

        outs.append(x)
        x = self.res_layer2(x)
        outs.append(x)
        x = self.res_layer3(x)
        outs.append(x)
        x = self.res_layer4(x)
        outs.append(x)
        return outs

    def _fusion_forward_features(self, x):

        # stage1 resnet
        res_outs = []
        res_outs_Y = []
        res_x = x
        res_x = self.conv1(res_x)
        res_x = self.bn1(res_x)
        res_x = self.relu(res_x)
        res_x = self.maxpool(res_x)
        res_x = self.res_layer1(res_x)

        res_x_Y = x
        res_x_Y = self.conv1_Y(res_x_Y)
        res_x_Y = self.bn1_Y(res_x_Y)
        res_x_Y = self.relu_Y(res_x_Y)
        res_x_Y = self.maxpool_Y(res_x_Y)
        res_x_Y = self.res_layer1_Y(res_x_Y)

        res_x, res_x_Y = self.cross_att_list[0](res_x, res_x_Y)
        res_outs_Y.append(res_x_Y)
        res_outs.append(res_x)


        # stage2 resnet
        res_x = self.res_layer2(res_x)
        res_x_Y = self.res_layer2_Y(res_x_Y)

        res_x, res_x_Y = self.cross_att_list[1](res_x, res_x_Y)
        res_outs_Y.append(res_x_Y)
        res_outs.append(res_x)



        # stage3 resnet
        res_x = self.res_layer3(res_x)
        res_x_Y = self.res_layer3_Y(res_x_Y)

        res_x, res_x_Y = self.cross_att_list[2](res_x, res_x_Y)
        res_outs_Y.append(res_x_Y)
        res_outs.append(res_x)


        # stage4 resnet
        res_x = self.res_layer4(res_x)
        res_x_Y = self.res_layer4_Y(res_x_Y)
        res_x, res_x_Y = self.cross_att_list[3](res_x, res_x_Y)
        res_outs_Y.append(res_x_Y)
        res_outs.append(res_x)

        return res_outs, res_outs_Y

    def _conver_features(self, res_features, trans_features):
        outs = []

        # for i, converge_layer in enumerate(self.converge_layer_list):
        #     out = converge_layer(torch.cat([res_features[i], trans_features[i]], dim=1))
        #     outs.append(out)
        #############################EncoderV3G8##########################
        for i, converge_layer in enumerate(self.converge_layer_list):
            out = converge_layer(res_features[i], trans_features[i])
            outs.append(out)
        ##################################################################
        return outs

    def forward(self, x1, x2):

        res_outs1 = self._res_forward_features(x1)
        res_outs2 = self._res_forward_features(x2)
        return res_outs1, res_outs2




class HyDecoder(nn.Module):
    def __init__(self, dims=[64, 128, 256, 512], nc=2):
        super().__init__()
        self.fusion_c1 = nn.Sequential(
            nn.Conv2d(in_channels=dims[0] * 2, out_channels=dims[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.fusion_c2 = nn.Sequential(
            nn.Conv2d(in_channels=dims[1] * 2, out_channels=dims[1], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[1]), nn.ReLU())
        self.fusion_c3 = nn.Sequential(
            nn.Conv2d(in_channels=dims[2] * 2, out_channels=dims[2], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[2]), nn.ReLU())
        self.fusion_c4 = nn.Sequential(
            nn.Conv2d(in_channels=dims[3] * 2, out_channels=dims[3], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[3]), nn.ReLU())
        self.up_c4 = nn.ConvTranspose2d(in_channels=dims[3], out_channels=dims[2], kernel_size=4, stride=2, padding=1)
        self.up_c3 = nn.ConvTranspose2d(in_channels=dims[2] * 2, out_channels=dims[1], kernel_size=4, stride=2,
                                        padding=1)
        self.up_c2 = nn.ConvTranspose2d(in_channels=dims[1] * 2, out_channels=dims[0], kernel_size=4, stride=2,
                                        padding=1)
        self.up_c1 = nn.ConvTranspose2d(in_channels=dims[0] * 2, out_channels=dims[0], kernel_size=4, stride=2,
                                        padding=1)

        self.up2input = nn.ConvTranspose2d(in_channels=dims[0], out_channels=dims[0], kernel_size=4, stride=2,
                                           padding=1)
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.final_predict = nn.Conv2d(in_channels=dims[0], out_channels=nc, kernel_size=3, padding=1)

    def forward(self, inputs1, inputs2):
        c1_1, c2_1, c3_1, c4_1 = inputs1
        c1_2, c2_2, c3_2, c4_2 = inputs2
        fusion_4 = self.fusion_c4(torch.cat([c4_1, c4_2], dim=1))
        up_fusion_4 = self.up_c4(fusion_4)

        fusion_3 = self.fusion_c3(torch.cat([c3_1, c3_2], dim=1))
        up_fusion_3 = self.up_c3(torch.cat([fusion_3, up_fusion_4], dim=1))
        fusion_2 = self.fusion_c2(torch.cat([c2_1, c2_2], dim=1))
        up_fusion_2 = self.up_c2(torch.cat([fusion_2, up_fusion_3], dim=1))
        fusion_1 = self.fusion_c1(torch.cat([c1_1, c1_2], dim=1))
        # H/2 x W/4
        up_fusion_1 = self.up_c1(torch.cat([fusion_1, up_fusion_2], dim=1))
        output = self.conv3x3_1(up_fusion_1)
        output = self.up2input(output)
        output = self.conv3x3_2(output)
        output = self.final_predict(output)
        return output


class _CrNonLocalBlock(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(_CrNonLocalBlock, self).__init__()

        self.in_channels = in_channels
        self.inter_channels = inter_channels

        if self.inter_channels is None:
            self.inter_channels = in_channels // 2
            if self.inter_channels == 0:
                self.inter_channels = 1

        conv = nn.Conv2d

        bn = nn.BatchNorm2d

        self.g1 = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0)
        self.g2 = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                       kernel_size=1, stride=1, padding=0)
        if bn_layer:
            self.W1 = nn.Sequential(
                conv(in_channels=self.inter_channels, out_channels=self.in_channels,
                     kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            self.W2 = nn.Sequential(
                conv(in_channels=self.inter_channels, out_channels=self.in_channels,
                     kernel_size=1, stride=1, padding=0),
                bn(self.in_channels)
            )
            nn.init.constant_(self.W1[1].weight, 0)
            nn.init.constant_(self.W1[1].bias, 0)
            nn.init.constant_(self.W2[1].weight, 0)
            nn.init.constant_(self.W2[1].bias, 0)
        else:
            self.W1 = conv(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
            self.W2 = conv(in_channels=self.inter_channels, out_channels=self.in_channels,
                           kernel_size=1, stride=1, padding=0)
            nn.init.constant_(self.W1.weight, 0)
            nn.init.constant_(self.W1.bias, 0)
            nn.init.constant_(self.W2.weight, 0)
            nn.init.constant_(self.W2.bias, 0)

        self.theta1 = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.theta2 = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                           kernel_size=1, stride=1, padding=0)
        self.phi1 = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.phi2 = conv(in_channels=self.in_channels, out_channels=self.inter_channels,
                         kernel_size=1, stride=1, padding=0)
        self.fuse_layer_x1 = nn.Sequential(ChannelAttention(self.in_channels * 2),
                                           nn.Conv2d(self.in_channels * 2, self.in_channels,
                                                     kernel_size=3, padding=1, stride=1),
                                           nn.BatchNorm2d(self.in_channels), nn.ReLU())
        self.fuse_layer_x2 = nn.Sequential(ChannelAttention(self.in_channels * 2),
                                           nn.Conv2d(self.in_channels * 2, self.in_channels,
                                                     kernel_size=3, padding=1, stride=1),
                                           nn.BatchNorm2d(self.in_channels), nn.ReLU())

    def forward(self, x1, x2):
        '''
        :param x: (b, c, t, h, w)
        :return:
        '''

        batch_size = x1.size(0)

        g_x1 = self.g1(x1).view(batch_size, self.inter_channels, -1)
        g_x1 = g_x1.permute(0, 2, 1)
        g_x2 = self.g2(x2).view(batch_size, self.inter_channels, -1)
        g_x2 = g_x2.permute(0, 2, 1)

        theta_x1 = self.theta1(x1).view(batch_size, self.inter_channels, -1)
        theta_x1 = theta_x1.permute(0, 2, 1)
        theta_x2 = self.theta2(x2).view(batch_size, self.inter_channels, -1)
        theta_x2 = theta_x2.permute(0, 2, 1)

        phi_x1 = self.phi1(x1).view(batch_size, self.inter_channels, -1)
        phi_x2 = self.phi2(x2).view(batch_size, self.inter_channels, -1)
        f1 = torch.matmul(theta_x1, phi_x2)
        f2 = torch.matmul(theta_x2, phi_x1)
        f_div_C1 = F.softmax(f1, dim=-1)
        f_div_C2 = F.softmax(f2, dim=-1)

        y1 = torch.matmul(f_div_C1, g_x2)
        y2 = torch.matmul(f_div_C2, g_x1)
        y1 = y1.permute(0, 2, 1).contiguous()
        y2 = y2.permute(0, 2, 1).contiguous()
        y1 = y1.view(batch_size, self.inter_channels, *x1.size()[2:])
        y2 = y2.view(batch_size, self.inter_channels, *x1.size()[2:])
        W_y1 = self.W1(y1)
        W_y2 = self.W2(y2)
        ##############EncoderV3G2#############
        # z1 = self.fuse_layer_x1(torch.cat([W_y2, x1], dim=1))
        # z2 = self.fuse_layer_x2(torch.cat([W_y1, x2], dim=1))
        #####################################
        ##############EncoderV3G3#############
        z1 = self.fuse_layer_x1(torch.cat([W_y1, x1], dim=1))
        z2 = self.fuse_layer_x2(torch.cat([W_y2, x2], dim=1))
        #####################################
        # z1 = W_y2 + x1
        # z2 = W_y1 + x2
        # z = W_y1 + x1

        # return torch.cat([z1, z2], dim=1)
        return z1, z2


class HyDecoderV2(nn.Module):
    def __init__(self, dims=[64, 128, 256, 512], nc=2):
        super().__init__()
        self.cross_att_c1 = _CrNonLocalBlock(in_channels=dims[0], inter_channels=dims[0])
        self.cross_att_c2 = _CrNonLocalBlock(in_channels=dims[1], inter_channels=dims[1])
        self.cross_att_c3 = _CrNonLocalBlock(in_channels=dims[2], inter_channels=dims[2])
        self.cross_att_c4 = _CrNonLocalBlock(in_channels=dims[3], inter_channels=dims[3])
        self.fusion_c1 = nn.Sequential(
            nn.Conv2d(in_channels=dims[0] * 2, out_channels=dims[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.fusion_c2 = nn.Sequential(
            nn.Conv2d(in_channels=dims[1] * 2, out_channels=dims[1], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[1]), nn.ReLU())
        self.fusion_c3 = nn.Sequential(
            nn.Conv2d(in_channels=dims[2] * 2, out_channels=dims[2], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[2]), nn.ReLU())
        self.fusion_c4 = nn.Sequential(
            nn.Conv2d(in_channels=dims[3] * 2, out_channels=dims[3], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[3]), nn.ReLU())
        self.up_c4 = nn.ConvTranspose2d(in_channels=dims[3], out_channels=dims[2], kernel_size=4, stride=2, padding=1)
        self.up_c3 = nn.ConvTranspose2d(in_channels=dims[2] * 2, out_channels=dims[1], kernel_size=4, stride=2,
                                        padding=1)
        self.up_c2 = nn.ConvTranspose2d(in_channels=dims[1] * 2, out_channels=dims[0], kernel_size=4, stride=2,
                                        padding=1)
        self.up_c1 = nn.ConvTranspose2d(in_channels=dims[0] * 2, out_channels=dims[0], kernel_size=4, stride=2,
                                        padding=1)

        self.up2input = nn.ConvTranspose2d(in_channels=dims[0], out_channels=dims[0], kernel_size=4, stride=2,
                                           padding=1)
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.final_predict = nn.Conv2d(in_channels=dims[0], out_channels=nc, kernel_size=3, padding=1)

    def forward(self, inputs1, inputs2):
        c1_1, c2_1, c3_1, c4_1 = inputs1
        c1_2, c2_2, c3_2, c4_2 = inputs2
        fusion_4 = self.fusion_c4(self.cross_att_c4(c4_1, c4_2))
        up_fusion_4 = self.up_c4(fusion_4)

        fusion_3 = self.fusion_c3(self.cross_att_c3(c3_1, c3_2))
        up_fusion_3 = self.up_c3(torch.cat([fusion_3, up_fusion_4], dim=1))
        fusion_2 = self.fusion_c2(self.cross_att_c2(c2_1, c2_2))
        up_fusion_2 = self.up_c2(torch.cat([fusion_2, up_fusion_3], dim=1))
        fusion_1 = self.fusion_c1(self.cross_att_c1(c1_1, c1_2))
        # H/2 x W/4
        up_fusion_1 = self.up_c1(torch.cat([fusion_1, up_fusion_2], dim=1))
        output = self.conv3x3_1(up_fusion_1)
        output = self.up2input(output)
        output = self.conv3x3_2(output)
        output = self.final_predict(output)
        return output

class HyDecoderV3(nn.Module):
    def __init__(self, dims=[64, 128, 256, 512], nc=2):
        super().__init__()
        self.fusion_c1 = BitFusion(in_channels=dims[0])
        self.fusion_c2 = BitFusion(in_channels=dims[1])
        self.fusion_c3 = BitFusion(in_channels=dims[2])
        self.fusion_c4 = BitFusion(in_channels=dims[3])
        self.up_c4 = nn.ConvTranspose2d(in_channels=dims[3], out_channels=dims[2], kernel_size=4, stride=2, padding=1)
        self.up_c3 = nn.ConvTranspose2d(in_channels=dims[2] * 2, out_channels=dims[1], kernel_size=4, stride=2,
                                        padding=1)
        self.up_c2 = nn.ConvTranspose2d(in_channels=dims[1] * 2, out_channels=dims[0], kernel_size=4, stride=2,
                                        padding=1)
        self.up_c1 = nn.ConvTranspose2d(in_channels=dims[0] * 2, out_channels=dims[0], kernel_size=4, stride=2,
                                        padding=1)

        self.up2input = nn.ConvTranspose2d(in_channels=dims[0], out_channels=dims[0], kernel_size=4, stride=2,
                                           padding=1)
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels=dims[0], out_channels=dims[0], kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(dims[0]), nn.ReLU())
        self.final_predict = nn.Conv2d(in_channels=dims[0], out_channels=nc, kernel_size=3, padding=1)

    def forward(self, inputs1, inputs2):
        c1_1, c2_1, c3_1, c4_1 = inputs1
        c1_2, c2_2, c3_2, c4_2 = inputs2
        fusion_4 = self.fusion_c4(c4_1, c4_2)
        up_fusion_4 = self.up_c4(fusion_4)

        fusion_3 = self.fusion_c3(c3_1, c3_2)
        up_fusion_3 = self.up_c3(torch.cat([fusion_3, up_fusion_4], dim=1))
        fusion_2 = self.fusion_c2(c2_1, c2_2)
        up_fusion_2 = self.up_c2(torch.cat([fusion_2, up_fusion_3], dim=1))
        fusion_1 = self.fusion_c1(c1_1, c1_2)
        # H/2 x W/4
        up_fusion_1 = self.up_c1(torch.cat([fusion_1, up_fusion_2], dim=1))
        output = self.conv3x3_1(up_fusion_1)
        output = self.up2input(output)
        output = self.conv3x3_2(output)
        output = self.final_predict(output)
        return output

class HybridModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.hybird_encoder = HybridEncoder(block=BasicBlock, layers=[2, 2, 2, 2])
        self.hybird_decoder = HyDecoder()

    def forward(self, x1, x2):
        inputs1, inputs2 = self.hybird_encoder(x1, x2)

        output = self.hybird_decoder(inputs1, inputs2)
        return output


class HybridModelV2(nn.Module):
    def __init__(self):
        super().__init__()
        self.hybird_encoder = HybridEncoder_SCM(block=BasicBlock, layers=[2, 2, 2, 2])
        self.hybird_decoder = HyDecoder()

    def forward(self, x1, x2):
        inputs1, inputs2 = self.hybird_encoder(x1, x2)

        output = self.hybird_decoder(inputs1, inputs2)
        return output

class HybridModel_SCM_CTFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.hybird_encoder = HybridEncoderV3(block=BasicBlock, layers=[2, 2, 2, 2])
        self.hybird_decoder = HyDecoder()

    def forward(self, x1, x2):
        inputs1, inputs2 = self.hybird_encoder(x1, x2)

        output = self.hybird_decoder(inputs1, inputs2)
        return output


class SingalModel_Trans_SCM_CTFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.hybird_encoder = HybridEncoderV3_SingalTrans(block=BasicBlock, layers=[2, 2, 2, 2])
        self.hybird_decoder = HyDecoderV3()

    def forward(self, x1, x2):
        inputs1, inputs2 = self.hybird_encoder(x1, x2)

        output = self.hybird_decoder(inputs1, inputs2)
        return output


class SingalModel_CNN_SCM_CTFF(nn.Module):
    def __init__(self):
        super().__init__()
        self.hybird_encoder = HybridEncoderV3_SingalCNN(block=BasicBlock, layers=[2, 2, 2, 2])
        self.hybird_decoder = HyDecoderV3()

    def forward(self, x1, x2):
        inputs1, inputs2 = self.hybird_encoder(x1, x2)

        output = self.hybird_decoder(inputs1, inputs2)
        return output

class HybridModelV3(nn.Module):
    def __init__(self):
        super().__init__()
        self.hybird_encoder = HybridEncoderV3(block=BasicBlock, layers=[2, 2, 2, 2])
        self.hybird_decoder = HyDecoderV3()

    def forward(self, x1, x2):
        inputs1, inputs2 = self.hybird_encoder(x1, x2)

        output = self.hybird_decoder(inputs1, inputs2)
        return output