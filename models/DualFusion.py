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


class LearnableCoefficient(nn.Module):
    def __init__(self):
        super(LearnableCoefficient, self).__init__()
        self.bias = nn.Parameter(torch.FloatTensor([1.0]), requires_grad=True)

    def forward(self, x):
        out = x * self.bias
        return out


class DWConv(nn.Module):
    def __init__(self, dim=768):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).view(B, C, H, W).contiguous()
        x = self.dwconv(x)
        x = x.flatten(2).transpose(1, 2)

        return x

class DepthwiseSeparableConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DepthwiseSeparableConv, self).__init__()
        self.depthwise = nn.Conv2d(in_channels, in_channels, kernel_size=3, padding=1, groups=in_channels)
        self.pointwise = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        return x

class ConvolutionalGLU(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        hidden_features = int(2 * hidden_features / 3)

        self.fc1 = nn.Conv2d(in_features, hidden_features * 2,kernel_size=1)

        self.dwconv = DepthwiseSeparableConv(hidden_features,hidden_features)
        self.act = act_layer()

        self.fc2 = nn.Conv2d(hidden_features, out_features,kernel_size=1)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x, v = self.fc1(x).chunk(2, dim=1)

        x = self.act(self.dwconv(x)) * v
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class Wide_Focus(nn.Module):


    def __init__(self,
                 in_channels,
                 out_channels):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same")
        self.conv2 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=2)
        self.conv3 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=3)
        self.conv4 = nn.Conv2d(in_channels, out_channels, 3, 1, padding="same", dilation=5)

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


class FusionModel(nn.Module):
    def __init__(self, in_channels, inter_channels=None, bn_layer=True):
        super(FusionModel, self).__init__()

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
        self.fuse_layer_x1 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_channels), nn.ReLU())
        self.fuse_layer_x2 = nn.Sequential(
            nn.Conv2d(self.in_channels, self.in_channels,
                      kernel_size=3, padding=1, stride=1),
            nn.BatchNorm2d(self.in_channels), nn.ReLU())
        self.wide_focus = Wide_Focus(self.in_channels * 2, self.in_channels * 2)
        self.conv_layer_1 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels * 2, kernel_size=3, stride=1,
                      padding=1, bias=False), nn.BatchNorm2d(self.in_channels * 2), nn.ReLU())
        self.conv_layer_2 = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels * 2, kernel_size=3, stride=1,
                      padding=1, bias=False), nn.BatchNorm2d(self.in_channels * 2), nn.ReLU())
        self.reduce_channel = nn.Sequential(
            nn.Conv2d(in_channels=self.in_channels * 2, out_channels=self.in_channels, kernel_size=1, stride=1,
                      padding=0, bias=False), nn.BatchNorm2d(self.in_channels), nn.ReLU())

    def forward(self, x1, x2):


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

        ##############EncoderV3G3#############
        z1 = self.fuse_layer_x1(torch.add(W_y2, x1))
        z2 = self.fuse_layer_x2(torch.add(W_y1, x2))
        out = torch.cat([z1, z2], dim=1)
        out = self.wide_focus(out)
        out = self.conv_layer_1(out)
        out = self.conv_layer_2(out)
        out = self.reduce_channel(out)


        return out


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
                                                num_heads=self.num_heads)  # num_heads=self.num_heads)
        self.attention2 = nn.MultiheadAttention(embed_dim=channels,
                                                bias=attention_bias,
                                                batch_first=True,
                                                # dropout = 0.0,
                                                num_heads=self.num_heads)

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

######################################################EncoderV3G9#####################################################
class FusionModelV2(nn.Module):
    def __init__(self,
                 # in_channels,
                 out_channels,
                 num_heads=8,
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

        self.cglu1 = ConvolutionalGLU(self.in_channels)
        self.cglu2 = ConvolutionalGLU(self.in_channels)

        self.fuse_layer = nn.Sequential(ChannelAttention(self.in_channels * 2),
                                        nn.Conv2d(self.in_channels * 2, self.in_channels,
                                                  kernel_size=1, padding=0, stride=1),
                                        nn.BatchNorm2d(self.in_channels), nn.ReLU())
        self.a1 = LearnableCoefficient()
        self.b1 = LearnableCoefficient()
        self.c1 = LearnableCoefficient()
        self.d1 = LearnableCoefficient()
        self.a2 = LearnableCoefficient()
        self.b2 = LearnableCoefficient()
        self.c2 = LearnableCoefficient()
        self.d2 = LearnableCoefficient()

    def forward(self, x_1, x_2):
        x1, x2 = self.attention_output(x_1, x_2)
        x1 = self.conv1(x1)
        x2 = self.conv2(x2)
        x1 = self.b1(x1)
        x_1 = self.a1(x_1)
        x2 = self.b2(x2)
        x_2 = self.a2(x_2)
        x1_sc = torch.add(x1, x_1)
        x2_sc = torch.add(x2, x_2)
        x1 = x1_sc.permute(0, 2, 3, 1)
        x2 = x2_sc.permute(0, 2, 3, 1)
        x1 = self.layernorm1(x1)
        x2 = self.layernorm2(x2)
        x1 = x1.permute(0, 3, 1, 2)
        x2 = x2.permute(0, 3, 1, 2)
        x1 = self.cglu1(x1)
        x2 = self.cglu2(x2)
        z1 = torch.add(self.d1(x1_sc), self.c1(x1))
        z2 = torch.add(self.d2(x2_sc), self.c2(x2))
        out = self.fuse_layer(torch.cat([z1, z2], dim=1))

        return out
