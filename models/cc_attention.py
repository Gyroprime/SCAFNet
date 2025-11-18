

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Softmax


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
        # return torch.cat([out * x1, out * x2], dim=1)
        return torch.add(out * x1, out * x2)


def INF(B, H, W):
    return -torch.diag(torch.tensor(float("inf")).cuda().repeat(H), 0).unsqueeze(0).repeat(B * W, 1, 1)


class CrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrissCrossAttention, self).__init__()
        self.query_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        m_batchsize, _, height, width = x.size()
        proj_query = self.query_conv(x)
        proj_query_H = proj_query.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0, 2,
                                                                                                                 1)
        proj_query_W = proj_query.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0, 2,
                                                                                                                 1)
        proj_key = self.key_conv(x)
        proj_key_H = proj_key.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W = proj_key.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value = self.value_conv(x)
        proj_value_H = proj_value.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W = proj_value.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H = (torch.bmm(proj_query_H, proj_key_H) + self.INF(m_batchsize, height, width)).view(m_batchsize, width,
                                                                                                     height,
                                                                                                     height).permute(0,
                                                                                                                     2,
                                                                                                                     1,
                                                                                                                     3)
        energy_W = torch.bmm(proj_query_W, proj_key_W).view(m_batchsize, height, width, width)
        concate = self.softmax(torch.cat([energy_H, energy_W], 3))

        att_H = concate[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height, height)

        att_W = concate[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H = torch.bmm(proj_value_H, att_H.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2, 3, 1)
        out_W = torch.bmm(proj_value_W, att_W.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2, 1, 3)

        return self.gamma * (out_H + out_W) + x


class CrossCrissCrossAttention(nn.Module):
    """ Criss-Cross Attention Module"""

    def __init__(self, in_dim):
        super(CrossCrissCrossAttention, self).__init__()
        self.in_channels = in_dim
        self.query_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.query_conv_2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.key_conv_2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 8, kernel_size=1)
        self.value_conv_1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.value_conv_2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)
        self.softmax = Softmax(dim=3)
        self.INF = INF
        self.gamma_1 = nn.Parameter(torch.zeros(1))
        self.gamma_2 = nn.Parameter(torch.zeros(1))
        self.shareca_1 = ShareChannelAttention(self.in_channels)
        self.shareca_2 = ShareChannelAttention(self.in_channels)

        ################################EncoderV3G7c3###############################
        self.act_1 = nn.Sequential(nn.BatchNorm2d(self.in_channels), nn.ReLU())
        self.act_2 = nn.Sequential(nn.BatchNorm2d(self.in_channels), nn.ReLU())

    def forward(self, x1, x2):
        m_batchsize, _, height, width = x1.size()
        proj_query_1 = self.query_conv_1(x1)
        proj_query_2 = self.query_conv_2(x2)
        proj_query_H_1 = proj_query_1.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0,
                                                                                                                     2,
                                                                                                                     1)
        proj_query_H_2 = proj_query_2.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height).permute(0,
                                                                                                                     2,
                                                                                                                     1)
        proj_query_W_1 = proj_query_1.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0,
                                                                                                                     2,
                                                                                                                     1)
        proj_query_W_2 = proj_query_2.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width).permute(0,
                                                                                                                     2,
                                                                                                                     1)

        proj_key_1 = self.key_conv_1(x1)
        proj_key_2 = self.key_conv_2(x2)
        proj_key_H_1 = proj_key_1.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_H_2 = proj_key_2.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_key_W_1 = proj_key_1.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_key_W_2 = proj_key_2.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value_1 = self.value_conv_1(x1)
        proj_value_2 = self.value_conv_2(x2)
        proj_value_H_1 = proj_value_1.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_H_2 = proj_value_2.permute(0, 3, 1, 2).contiguous().view(m_batchsize * width, -1, height)
        proj_value_W_1 = proj_value_1.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        proj_value_W_2 = proj_value_2.permute(0, 2, 1, 3).contiguous().view(m_batchsize * height, -1, width)
        energy_H_1 = (torch.bmm(proj_query_H_2, proj_key_H_1) + self.INF(m_batchsize, height, width)).view(m_batchsize,
                                                                                                           width,
                                                                                                           height,
                                                                                                           height).permute(
            0, 2, 1, 3)
        energy_H_2 = (torch.bmm(proj_query_H_1, proj_key_H_2) + self.INF(m_batchsize, height, width)).view(m_batchsize,
                                                                                                           width,
                                                                                                           height,
                                                                                                           height).permute(
            0, 2, 1, 3)
        energy_W_1 = torch.bmm(proj_query_W_2, proj_key_W_1).view(m_batchsize, height, width, width)
        energy_W_2 = torch.bmm(proj_query_W_1, proj_key_W_2).view(m_batchsize, height, width, width)
        concate_1 = self.softmax(torch.cat([energy_H_1, energy_W_1], 3))
        concate_2 = self.softmax(torch.cat([energy_H_2, energy_W_2], 3))

        att_H_1 = concate_1[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
                                                                                     height)
        att_H_2 = concate_2[:, :, :, 0:height].permute(0, 2, 1, 3).contiguous().view(m_batchsize * width, height,
                                                                                     height)

        att_W_1 = concate_1[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        att_W_2 = concate_2[:, :, :, height:height + width].contiguous().view(m_batchsize * height, width, width)
        out_H_1 = torch.bmm(proj_value_H_1, att_H_1.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2,
                                                                                                                   3, 1)
        out_H_2 = torch.bmm(proj_value_H_2, att_H_2.permute(0, 2, 1)).view(m_batchsize, width, -1, height).permute(0, 2,
                                                                                                                   3, 1)
        out_W_1 = torch.bmm(proj_value_W_1, att_W_1.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2,
                                                                                                                   1, 3)
        out_W_2 = torch.bmm(proj_value_W_2, att_W_2.permute(0, 2, 1)).view(m_batchsize, height, -1, width).permute(0, 2,
                                                                                                                   1, 3)

        #####################EncoderV3G71################c3#c5
        x_1 = self.shareca_1(self.gamma_2 * (out_H_2 + out_W_2) + x2, x1)
        x_2 = self.shareca_2(self.gamma_1 * (out_H_1 + out_W_1) + x1, x2)
        #####################EncoderV3G7c1###############c4#c6
        # x_1 = self.shareca_1(self.gamma_1 * (out_H_1 + out_W_1) + x1, x1)
        # x_2 = self.shareca_2(self.gamma_2 * (out_H_2 + out_W_2) + x2, x2)
        #####################EncoderV3G7c2###############
        # z1 = self.shareca_1(self.gamma_1 * (out_H_1 + out_W_1) + x1, x1)
        # z2 = self.shareca_2(self.gamma_2 * (out_H_2 + out_W_2) + x2, x2)
        ####################################################
        # z1 = self.fuse_layer_x1(x_1)
        # z2 = self.fuse_layer_x2(x_2)
        #####################################################
        # return self.gamma_1*(out_H_1 + out_W_1) + x1, self.gamma_2*(out_H_2 + out_W_2) + x2
        ################################EncoderV3G7c3###############################
        out1 = self.act_1(x_1 + x1)
        out2 = self.act_2(x_2 + x2)
        ################################EncoderV3G7c5 c6 Channel Exchange############################
        exchange_mask_1 = torch.arange(self.in_channels) % 2 == 0
        exchange_mask_2 = torch.arange(self.in_channels) % 2 == 0
        out_1, out_2 = torch.zeros_like(x1), torch.zeros_like(x2)
        exchange_mask_1 = exchange_mask_1.unsqueeze(0).expand((m_batchsize, -1))
        exchange_mask_2 = exchange_mask_2.unsqueeze(0).expand((m_batchsize, -1))
        out_1[~exchange_mask_1,] = out1[~exchange_mask_1,]
        out_2[~exchange_mask_2,] = out2[~exchange_mask_2,]
        out_1[exchange_mask_1,] = x1[exchange_mask_1,]
        out_2[exchange_mask_2,] = x2[exchange_mask_2,]
        # return z1+x1, z2+x2
        return out_1, out_2


if __name__ == '__main__':
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = CrossCrissCrossAttention(64).to(device)
    x1 = torch.randn(2, 64, 64, 64).to(device)
    x2 = torch.randn(2, 64, 64, 64).to(device)
    out1, out2 = model(x1, x2)
    print(out1.shape, out2.shape)
