import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class ShareChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ShareChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1)
        self.fc2 = nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1)
        self.relu = nn.ReLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        avg_out = self.fc2(self.relu(self.fc1(self.avg_pool(torch.add(x1, x2)))))
        max_out = self.fc2(self.relu(self.fc1(self.max_pool(torch.add(x1, x2)))))
        out = self.sigmoid(avg_out + max_out)
        return torch.cat([out * x1, out * x2], dim=1)


class Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_q = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias,
                                groups=channels)
        self.layernorm_q = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_k = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                groups=channels)
        self.layernorm_v = nn.LayerNorm(channels, eps=1e-5)

        self.attention = nn.MultiheadAttention(embed_dim=channels,
                                               bias=attention_bias,
                                               batch_first=True,
                                               # dropout = 0.0,
                                               num_heads=1)

    def _build_projection(self, x, qkv):

        if qkv == "q":
            x1 = F.relu(self.conv_q(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k":
            x1 = F.relu(self.conv_k(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v":
            x1 = F.relu(self.conv_v(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x):
        q = self._build_projection(x, "q")
        k = self._build_projection(x, "k")
        v = self._build_projection(x, "v")

        return q, k, v

    def forward(self, x):
        q, k, v = self.forward_conv(x)
        q = q.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        k = k.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        v = v.view(x.shape[0], x.shape[1], x.shape[2] * x.shape[3])
        q = q.permute(0, 2, 1)
        k = k.permute(0, 2, 1)
        v = v.permute(0, 2, 1)
        x1 = self.attention(query=q, value=v, key=k, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        return x1


class Transformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 dpr,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()

        self.attention_output = Attention(channels=out_channels,
                                          num_heads=num_heads,
                                          proj_drop=proj_drop,
                                          padding_q=padding_q,
                                          padding_kv=padding_kv,
                                          stride_kv=stride_kv,
                                          stride_q=stride_q,
                                          attention_bias=attention_bias,
                                          )

        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus = Wide_Focus(out_channels, out_channels)

    def forward(self, x):
        x1 = self.attention_output(x)
        x1 = self.conv1(x1)
        x2 = torch.add(x1, x)
        x3 = x2.permute(0, 2, 3, 1)
        x3 = self.layernorm(x3)
        x3 = x3.permute(0, 3, 1, 2)
        x3 = self.wide_focus(x3)
        x3 = torch.add(x2, x3)
        return x3


class CTransformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 # dpr,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()

        self.attention_output = Cross_Attention(channels=out_channels,
                                                num_heads=num_heads,
                                                proj_drop=proj_drop,
                                                padding_q=padding_q,
                                                padding_kv=padding_kv,
                                                stride_kv=stride_kv,
                                                stride_q=stride_q,
                                                attention_bias=attention_bias,
                                                )
        self.in_channels = out_channels
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm1 = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.layernorm2 = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus1 = Wide_Focus(out_channels, out_channels)
        self.wide_focus2 = Wide_Focus(out_channels, out_channels)
        ###########################EncoderV3G5 ##########################################
        # self.fuse_layer_x1 = nn.Sequential(ChannelAttention(self.in_channels * 2),
        #                                    nn.Conv2d(self.in_channels * 2, self.in_channels,
        #                                              kernel_size=3, padding=1, stride=1),
        #                                    nn.BatchNorm2d(self.in_channels), nn.ReLU())
        # self.fuse_layer_x2 = nn.Sequential(ChannelAttention(self.in_channels * 2),
        #                                    nn.Conv2d(self.in_channels * 2, self.in_channels,
        #                                              kernel_size=3, padding=1, stride=1),
        #                                    nn.BatchNorm2d(self.in_channels), nn.ReLU())
        ###################################################################################################
        #############################################EncoderV3G6 ##########################################
        self.shareca_1 = ShareChannelAttention(self.in_channels)
        self.shareca_2 = ShareChannelAttention(self.in_channels)
        self.fuse_layer_x1 = nn.Sequential(
                                           nn.Conv2d(self.in_channels * 2, self.in_channels,
                                                     kernel_size=3, padding=1, stride=1),
                                           nn.BatchNorm2d(self.in_channels), nn.ReLU())
        self.fuse_layer_x2 = nn.Sequential(
                                           nn.Conv2d(self.in_channels * 2, self.in_channels,
                                                     kernel_size=3, padding=1, stride=1),
                                           nn.BatchNorm2d(self.in_channels), nn.ReLU())
        ###################################################################################################

    def forward(self, x_1, x_2):
        x1, x2 = self.attention_output(x_1, x_2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x1_sc = torch.add(x1, x_1)
        x2_sc = torch.add(x2, x_2)
        x1 = x1_sc.permute(0, 2, 3, 1)
        x2 = x2_sc.permute(0, 2, 3, 1)
        x1 = self.layernorm1(x1)
        x2 = self.layernorm2(x2)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x1 = self.wide_focus1(x1)
        x2 = self.wide_focus2(x2)
        x1 = torch.add(x1_sc, x1)
        x2 = torch.add(x2_sc, x2)
        #####################EncoderV3G5 #################
        # z1 = self.fuse_layer_x1(torch.cat([x1, x_1], dim=1))
        # z2 = self.fuse_layer_x2(torch.cat([x2, x_2], dim=1))
        #################################################
        #####################EncoderV3G6 #################
        x1 = self.shareca_1(x1,x_1)
        x2 = self.shareca_2(x2,x_2)
        z1 = self.fuse_layer_x1(x1)
        z2 = self.fuse_layer_x2(x2)
        ##################################################
        # return x1 + x_1, x2 + x_2
        return z1+x_1, z2+x_2


class Wide_Focus(nn.Module):
    """
    Wide-Focus module.
    """

    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")

    def forward(self, x):
        x1 = self.conv1(x)
        x1 = F.gelu(x1)
        x1 = F.dropout(x1, 0.1)
        x2 = self.conv2(x)
        x2 = F.gelu(x2)
        x2 = F.dropout(x2, 0.1)
        x3 = self.conv3(x)
        x3 = F.gelu(x3)
        x3 = F.dropout(x3, 0.1)
        added = torch.add(x1, x2)
        added = torch.add(added, x3)
        x_out = self.conv4(added)
        x_out = F.gelu(x_out)
        x_out = F.dropout(x_out, 0.1)
        return x_out


class Cross_Attention(nn.Module):
    def __init__(self,
                 channels,
                 num_heads,
                 proj_drop=0.0,
                 kernel_size=3,
                 stride_kv=1,
                 stride_q=1,
                 padding_kv="same",
                 padding_q="same",
                 attention_bias=True
                 ):
        super().__init__()
        self.stride_kv = stride_kv
        self.stride_q = stride_q
        self.num_heads = num_heads
        self.proj_drop = proj_drop

        self.conv_q1 = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias,
                                 groups=channels)
        self.layernorm_q1 = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k1 = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                 groups=channels)
        self.layernorm_k1 = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v1 = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                 groups=channels)
        self.layernorm_v1 = nn.LayerNorm(channels, eps=1e-5)

        self.conv_q2 = nn.Conv2d(channels, channels, kernel_size, stride_q, padding_q, bias=attention_bias,
                                 groups=channels)
        self.layernorm_q2 = nn.LayerNorm(channels, eps=1e-5)
        self.conv_k2 = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                 groups=channels)
        self.layernorm_k2 = nn.LayerNorm(channels, eps=1e-5)
        self.conv_v2 = nn.Conv2d(channels, channels, kernel_size, stride_kv, stride_kv, bias=attention_bias,
                                 groups=channels)
        self.layernorm_v2 = nn.LayerNorm(channels, eps=1e-5)

        self.attention1 = nn.MultiheadAttention(embed_dim=channels,
                                                bias=attention_bias,
                                                batch_first=True,
                                                # dropout = 0.0,
                                                num_heads=1)  # num_heads=self.num_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim=channels,
                                                bias=attention_bias,
                                                batch_first=True,
                                                # dropout = 0.0,
                                                num_heads=1)

    def _build_projection(self, x, qkv):

        if qkv == "q1":
            x1 = F.relu(self.conv_q1(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q1(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k1":
            x1 = F.relu(self.conv_k1(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k1(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v1":
            x1 = F.relu(self.conv_v1(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v1(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "q2":
            x1 = F.relu(self.conv_q2(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_q2(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "k2":
            x1 = F.relu(self.conv_k2(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_k2(x1)
            proj = x1.permute(0, 3, 1, 2)
        elif qkv == "v2":
            x1 = F.relu(self.conv_v2(x))
            x1 = x1.permute(0, 2, 3, 1)
            x1 = self.layernorm_v2(x1)
            proj = x1.permute(0, 3, 1, 2)

        return proj

    def forward_conv(self, x, branch):
        if branch == 1:
            q = self._build_projection(x, "q1")
            k = self._build_projection(x, "k1")
            v = self._build_projection(x, "v1")
        elif branch == 2:
            q = self._build_projection(x, "q2")
            k = self._build_projection(x, "k2")
            v = self._build_projection(x, "v2")
        return q, k, v

    def forward(self, x1, x2):
        res = x1
        trans = x2
        q1, k1, v1 = self.forward_conv(x1, 1)
        q2, k2, v2 = self.forward_conv(x2, 2)

        q1 = q1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3])
        k1 = k1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3])
        v1 = v1.view(x1.shape[0], x1.shape[1], x1.shape[2] * x1.shape[3])
        q1 = q1.permute(0, 2, 1)
        k1 = k1.permute(0, 2, 1)
        v1 = v1.permute(0, 2, 1)

        q2 = q2.view(x2.shape[0], x2.shape[1], x2.shape[2] * x2.shape[3])
        k2 = k2.view(x2.shape[0], x2.shape[1], x2.shape[2] * x2.shape[3])
        v2 = v2.view(x2.shape[0], x2.shape[1], x2.shape[2] * x2.shape[3])
        q2 = q2.permute(0, 2, 1)
        k2 = k2.permute(0, 2, 1)
        v2 = v2.permute(0, 2, 1)

        x1 = self.attention1(query=q1, value=v2, key=k2, need_weights=False)
        x2 = self.attention2(query=q2, value=v1, key=k1, need_weights=False)

        x1 = x1[0].permute(0, 2, 1)
        x1 = x1.view(x1.shape[0], x1.shape[1], np.sqrt(x1.shape[2]).astype(int), np.sqrt(x1.shape[2]).astype(int))
        x1 = F.dropout(x1, self.proj_drop)

        x2 = x2[0].permute(0, 2, 1)
        x2 = x2.view(x2.shape[0], x2.shape[1], np.sqrt(x2.shape[2]).astype(int), np.sqrt(x2.shape[2]).astype(int))
        x2 = F.dropout(x2, self.proj_drop)
        return x1, x2

class CTransformer(nn.Module):

    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads,
                 # dpr,
                 proj_drop=0.0,
                 attention_bias=True,
                 padding_q="same",
                 padding_kv="same",
                 stride_kv=1,
                 stride_q=1):
        super().__init__()

        self.attention_output = Cross_Attention(channels=out_channels,
                                                num_heads=num_heads,
                                                proj_drop=proj_drop,
                                                padding_q=padding_q,
                                                padding_kv=padding_kv,
                                                stride_kv=stride_kv,
                                                stride_q=stride_q,
                                                attention_bias=attention_bias,
                                                )
        self.in_channels = out_channels
        self.conv1 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, padding="same")
        self.layernorm1 = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.layernorm2 = nn.LayerNorm(self.conv1.out_channels, eps=1e-5)
        self.wide_focus1 = Wide_Focus(out_channels, out_channels)
        self.wide_focus2 = Wide_Focus(out_channels, out_channels)
        ###########################EncoderV3G5 ##########################################
        # self.fuse_layer_x1 = nn.Sequential(ChannelAttention(self.in_channels * 2),
        #                                    nn.Conv2d(self.in_channels * 2, self.in_channels,
        #                                              kernel_size=3, padding=1, stride=1),
        #                                    nn.BatchNorm2d(self.in_channels), nn.ReLU())
        # self.fuse_layer_x2 = nn.Sequential(ChannelAttention(self.in_channels * 2),
        #                                    nn.Conv2d(self.in_channels * 2, self.in_channels,
        #                                              kernel_size=3, padding=1, stride=1),
        #                                    nn.BatchNorm2d(self.in_channels), nn.ReLU())
        ###################################################################################################
        #############################################EncoderV3G6 ##########################################
        self.shareca_1 = ShareChannelAttention(self.in_channels)
        self.shareca_2 = ShareChannelAttention(self.in_channels)
        self.fuse_layer_x1 = nn.Sequential(
                                           nn.Conv2d(self.in_channels * 2, self.in_channels,
                                                     kernel_size=3, padding=1, stride=1),
                                           nn.BatchNorm2d(self.in_channels), nn.ReLU())
        self.fuse_layer_x2 = nn.Sequential(
                                           nn.Conv2d(self.in_channels * 2, self.in_channels,
                                                     kernel_size=3, padding=1, stride=1),
                                           nn.BatchNorm2d(self.in_channels), nn.ReLU())
        ###################################################################################################

    def forward(self, x_1, x_2):
        x1, x2 = self.attention_output(x_1, x_2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x1_sc = torch.add(x1, x_1)
        x2_sc = torch.add(x2, x_2)
        x1 = x1_sc.permute(0, 2, 3, 1)
        x2 = x2_sc.permute(0, 2, 3, 1)
        x1 = self.layernorm1(x1)
        x2 = self.layernorm2(x2)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x1 = self.wide_focus1(x1)
        x2 = self.wide_focus2(x2)
        x1 = torch.add(x1_sc, x1)
        x2 = torch.add(x2_sc, x2)
        #####################EncoderV3G5 #################
        # z1 = self.fuse_layer_x1(torch.cat([x1, x_1], dim=1))
        # z2 = self.fuse_layer_x2(torch.cat([x2, x_2], dim=1))
        #################################################
        #####################EncoderV3G6 #################
        x1 = self.shareca_1(x1,x_1)
        x2 = self.shareca_2(x2,x_2)
        z1 = self.fuse_layer_x1(x1)
        z2 = self.fuse_layer_x2(x2)
        ##################################################
        # return x1 + x_1, x2 + x_2
        return z1+x_1, z2+x_2