import torch
import torch.nn as nn
import torch.nn.functional as F

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        # self.act=SiLU()
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):

        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out * x


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


class BitFusion(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(BitFusion, self).__init__()
        self.gap_fc = nn.Sequential(
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Conv2d(in_channels * 2, in_channels * 2 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2 // reduction, in_channels, 1)
        )

        self.gmp_fc = nn.Sequential(
            nn.AdaptiveMaxPool2d((1, 1)),
            nn.Conv2d(in_channels * 2, in_channels * 2 // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels * 2 // reduction, in_channels, 1)
        )
        self.sigmoid = nn.Sigmoid()
        self.conv_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv1x1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels * 5, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv3x3_1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU())
        self.conv3x3_2 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels*2, out_channels=in_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(in_channels), nn.ReLU())
        self.GeLU = nn.GELU()

    def forward(self, x1, x2):
        gap_fc_out = self.gap_fc(torch.cat([x1, x2], dim=1))
        gmp_fc_out = self.gmp_fc(torch.cat([x1, x2], dim=1))
        gap_gmp_out = self.sigmoid(gap_fc_out + gmp_fc_out)
        x1_att = x1 * gap_gmp_out
        x2_att = x2 * gap_gmp_out
        x1_att = self.conv_1(x1_att)
        # x2_att = self.conv_1(x2_att)
        x2_att = self.conv_2(x2_att)
        mul = x1_att * x2_att
        cat_x1 = self.conv3x3_1(torch.cat([x1_att, mul], dim=1))
        cat_x2 = self.conv3x3_2(torch.cat([x2_att, mul], dim=1))

        out = self.sigmoid(self.conv1x1(torch.cat([mul, x1_att, x2_att, cat_x1, cat_x2], dim=1)))
        out = self.GeLU(out * x1_att + out * x2_att)
        return out

if __name__ == "__main__":
    tmp1 = torch.randn(1, 16, 256, 256)
    tmp2 = torch.randn(1, 16, 256, 256)
    # bifusion = BitFusion(3, 3).to(torch.device('cuda:0'))
    bifusion = BitFusion(16)
    out = bifusion(tmp1, tmp2)
    print(out.shape)
