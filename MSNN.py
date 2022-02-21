import torch.nn.functional as F
import torch.nn as nn
import math
import torch


# from HCD_CNN.Attention import SpatialAttention, ChannelAttention


class GELU(nn.Module):
    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(math.sqrt(2 / math.pi) * (x + 0.044715 * torch.pow(x, 3))))


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            GELU(),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y.expand_as(x)


class SEBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.se = SELayer(planes * 4, reduction)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)
        out = self.se(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class BasicBlock(nn.Module):
    def __init__(self, input, output):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(input, output, kernel_size=(2, 2), stride=1, padding=1, dilation=2),
                                   nn.BatchNorm2d(output),
                                   GELU(),
                                   nn.Dropout(0.1),

                                   nn.Conv2d(input, output, kernel_size=(2, 2), stride=1, padding=1, dilation=2),
                                   nn.BatchNorm2d(output),
                                   SELayer(output),
                                   GELU(),
                                   nn.Dropout(0.1))

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out += residual
        return out


class MSNN(nn.Module):
    def __init__(self, input_min_2D=25, input_max_2D=75, input_min_3D=9, input_max_3D=31, output_min_2D=1,
                 output_max_2D=1,
                 output_min_3D=1, output_max_3D=1):
        super(MSNN, self).__init__()
        self.input_min_2D, self.input_max_2D = input_min_2D, input_max_2D
        self.input_min_3D, self.input_max_3D = input_min_3D, input_max_3D

        self.output_min_2D, self.output_max_2D = output_min_2D, output_max_2D
        self.output_min_3D, self.output_max_3D = output_min_3D, output_max_3D

        self.Fea_Min_2D_CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(3, 2), stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(8),
            GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(3, 2), stride=1, dilation=2, bias=False),
            nn.BatchNorm2d(8),
            GELU(),
            nn.MaxPool2d(kernel_size=(2, 2), stride=1),
            nn.Dropout(0.1),

            BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8),
            BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8),
            BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8)
        )

        self.Fea_Max_2D_CNN = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=8, kernel_size=(5, 3), stride=1, dilation=3, bias=False),
            nn.BatchNorm2d(8),
            GELU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(7, 3), stride=1, dilation=3, bias=False),
            nn.BatchNorm2d(8),
            GELU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=8, out_channels=8, kernel_size=(9, 3), stride=1, dilation=3, bias=False),
            nn.BatchNorm2d(8),
            GELU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1),
            nn.Dropout(0.1),

            BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8),
            BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8),
            BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8), BasicBlock(8, 8),
        )

        self.Fea_Min_3D_CNN = nn.Sequential(
            nn.Conv2d(in_channels=9, out_channels=16, kernel_size=(3, 2), bias=False),
            nn.BatchNorm2d(16),
            GELU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=16, out_channels=16, kernel_size=(5, 2), bias=False),
            nn.BatchNorm2d(16),
            GELU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1),
            nn.Dropout(0.1),

            BasicBlock(16, 16), BasicBlock(16, 16), BasicBlock(16, 16), BasicBlock(16, 16),
            BasicBlock(16, 16), BasicBlock(16, 16), BasicBlock(16, 16), BasicBlock(16, 16),
            BasicBlock(16, 16), BasicBlock(16, 16), BasicBlock(16, 16), BasicBlock(16, 16),
        )

        self.Fea_Max_3D_CNN = nn.Sequential(
            nn.Conv2d(in_channels=31, out_channels=64, kernel_size=(3, 2), bias=False),
            nn.BatchNorm2d(64),
            GELU(),
            nn.MaxPool2d(kernel_size=(3, 3), stride=1),
            nn.Dropout(0.1),

            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(5, 2), bias=False),
            nn.BatchNorm2d(64),
            GELU(),
            nn.AvgPool2d(kernel_size=(3, 3), stride=1),
            nn.Dropout(0.1),

            BasicBlock(64, 64), BasicBlock(64, 64), BasicBlock(64, 64), BasicBlock(64, 64),
            BasicBlock(64, 64), BasicBlock(64, 64), BasicBlock(64, 64), BasicBlock(64, 64),
            BasicBlock(64, 64), BasicBlock(64, 64), BasicBlock(64, 64), BasicBlock(64, 64),
        )

        self.Fea_fusion_3D_CNN = nn.Sequential(

            BasicBlock(96, 96), BasicBlock(96, 96), BasicBlock(96, 96), BasicBlock(96, 96),
            BasicBlock(96, 96), BasicBlock(96, 96), BasicBlock(96, 96), BasicBlock(96, 96),
            BasicBlock(96, 96), BasicBlock(96, 96), BasicBlock(96, 96), BasicBlock(96, 96),
            BasicBlock(96, 96), BasicBlock(96, 96), BasicBlock(96, 96), BasicBlock(96, 96),

            nn.Conv2d(96, 64, kernel_size=(2, 2), stride=1, padding=1, dilation=2),
            nn.BatchNorm2d(64),
            GELU(),
            nn.Dropout(0.1),

            nn.Conv2d(64, 32, kernel_size=(2, 2), stride=1, padding=1, dilation=2),
            nn.BatchNorm2d(32),
            GELU(),
            nn.Dropout(0.1),

            nn.Conv2d(32, 16, kernel_size=(2, 2), stride=1, padding=1, dilation=2),
            nn.BatchNorm2d(16),
            GELU(),
            nn.Dropout(0.1),

            nn.Conv2d(16, 1, kernel_size=(2, 2), stride=1, padding=1, dilation=2),
            nn.BatchNorm2d(1),
            GELU(),
            nn.Dropout(0.1))

        self.fc = nn.Sequential(nn.Linear(45, 1),
                                nn.Sigmoid())

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)
            elif isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight.data, 0, 0.01)
                if m.bias is not None:
                    m.bias.data.fill_(0)

    def forward(self, fea_min_2D, fea_max_2D, fea_min_3D, fea_max_3D):

        out_min_2D = self.Fea_Min_2D_CNN(fea_min_2D)
        # print(out_min_2D.shape)
        out_max_2D = self.Fea_Max_2D_CNN(fea_max_2D)
        # print(out_max_2D.shape)
        out_min_3D = self.Fea_Min_3D_CNN(fea_min_3D)
        # print(out_min_3D.shape)
        out_max_3D = self.Fea_Max_3D_CNN(fea_max_3D)
        # print(out_max_3D.shape)

        out_fusion_3D = torch.cat((out_min_2D, out_max_2D), 1)
        out_fusion_3D = torch.cat((out_fusion_3D, out_min_3D), 1)
        out_fusion_3D = torch.cat((out_fusion_3D, out_max_3D), 1)
        # print(out_fusion_3D.shape)
        out = self.Fea_fusion_3D_CNN(out_fusion_3D)
        out = torch.squeeze(torch.squeeze(out, 0), 0)
        out = torch.reshape(out, (-1, 45))

        out = self.fc(out)
        # print(out)
        return out

# if __name__ == '__main__':
#     a = torch.randn(1, 1, 25, 9)
#     b = torch.randn(1, 1, 75, 27)
#     c = torch.randn(1, 9, 25, 9)
#     d = torch.randn(1, 31, 25, 9)
#
#     Hybrid_Cascaded_Dilated_CCN = Hybrid_Cascaded_Dilated_CCN_Framework()
#     out = Hybrid_Cascaded_Dilated_CCN(a, b, c, d)
