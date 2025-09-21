import torch
import torch.nn as nn
import math
import torchvision
from functools import partial
import torch.nn.functional as F

def conv3x3(in_planes, out_planes, stride=1):
    "3x3 convolution with padding"
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc2(self.relu1(self.fc1(self.avg_pool(x))))
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = avg_out + max_out
        return self.sigmoid(out)


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


# decoding block used in PCFN
class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)

        self.ca = ChannelAttention(inplanes)
        self.sa = SpatialAttention()

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = x

        out = self.ca(x) * x
        out = self.conv1(out)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        # out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(planes * 4)
        self.sa = SpatialAttention()

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

        out = self.ca(out) * out
        out = self.sa(out) * out

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out



class Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(Block, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, in_planes // 2, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        if (kernel == 1):
            pad = 0
        else:
            pad = 1
        self.conv2 = nn.Conv2d(in_planes // 2, out_planes, kernel_size=kernel, stride=stride,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(in_planes // 2)
        self.bn2 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        conv1 = self.conv1(input)
        bn1 = self.relu(self.bn1(conv1))

        conv2 = self.conv2(bn1)
        out = self.relu(self.bn2(conv2))

        return out


class Out_Block(nn.Module):
    def __init__(self, in_planes, out_planes, kernel=3, stride=1):
        super(Out_Block, self).__init__()

        if (kernel == 1):
            pad = 0
        else:
            pad = 1

        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=kernel, stride=stride,
                               padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, input):
        conv1 = self.conv1(input)
        bn1 = self.bn1(conv1)

        return self.relu(bn1)


# Feature Difference Enhancement (FDE)
class FDE(nn.Module):
    def __init__(self, kernel_size=7):
        super(FDE, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=padding)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x1, x2):
        feat = torch.cat([x1, x2], 1)
        x = torch.abs(x2 - x1)

        avg_out = torch.mean(x, axis=1, keepdim=1)
        max_out = torch.max(x, axis=1, keepdim=1)

        x = torch.concat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        x = self.sigmoid(x)
        out = feat * x + feat
        return out


nonlinearity = partial(F.relu, inplace=True)


class Resnet(nn.Module):
    def __init__(self):
        super(Resnet, self).__init__()
        resnet = torchvision.models.resnet34(pretrained=True)
        # resnet.conv1.in_channels =6

        self.firstconv = resnet.conv1
        # self.firstconv = nn.Conv2d(4, 64, kernel_size=(7, 7), stride=(2, 2), padding=3, bias=False)
        self.firstbn = resnet.bn1
        self.firstrelu = resnet.relu
        self.firstmaxpool = resnet.maxpool
        self.encoder1 = resnet.layer1
        self.encoder2 = resnet.layer2
        self.encoder3 = resnet.layer3
        self.encoder4 = resnet.layer4

    def forward(self, x):
        # Encoder
        x = self.firstconv(x)
        x = self.firstbn(x)
        x = self.firstrelu(x)
        e0 = self.firstmaxpool(x)
        e1 = self.encoder1(e0)
        e2 = self.encoder2(e1)
        e3 = self.encoder3(e2)
        e4 = self.encoder4(e3)

        return [x, e1, e2, e3, e4]


class Dblock(nn.Module):
    def __init__(self, channel):
        super(Dblock, self).__init__()
        self.dilate1 = nn.Conv2d(channel, channel, kernel_size=3, dilation=1, padding=1)
        self.dilate2 = nn.Conv2d(channel, channel, kernel_size=3, dilation=2, padding=2)
        self.dilate3 = nn.Conv2d(channel, channel, kernel_size=3, dilation=4, padding=4)
        self.dilate4 = nn.Conv2d(channel, channel, kernel_size=3, dilation=8, padding=8)
        # self.dilate5 = nn.Conv2d(channel, channel, kernel_size=3, dilation=16, padding=16)
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        dilate1_out = nonlinearity(self.dilate1(x))
        dilate2_out = nonlinearity(self.dilate2(dilate1_out))
        dilate3_out = nonlinearity(self.dilate3(dilate2_out))
        dilate4_out = nonlinearity(self.dilate4(dilate3_out))
        # dilate5_out = nonlinearity(self.dilate5(dilate4_out))
        out = x + dilate1_out + dilate2_out + dilate3_out + dilate4_out  # + dilate5_out
        return out


# PCFN pytorch version
class PCFN(nn.Module):
    def __init__(self,LCM_class_num = 6, SCD_class_num =1):
        # backbone encoder
        super(PCFN, self).__init__()
        self.resnet = Resnet()
        self.SCD_class_num = SCD_class_num
        self.LCM_class_num = LCM_class_num

        # Siamese Decoders
        self.CBAM1_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM1_2 = BasicBlock(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM1_3 = BasicBlock(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM1_4 = BasicBlock(inplanes=2 * 64 + 256, planes=128, stride=1)
        self.CBAM1_5 = BasicBlock(inplanes=2 * 64 + 128, planes=64, stride=1)

        self.CBAM2_1 = BasicBlock(inplanes=2 * 512, planes=512, stride=1)
        self.CBAM2_2 = BasicBlock(inplanes=2 * 256 + 512, planes=512, stride=1)
        self.CBAM2_3 = BasicBlock(inplanes=2 * 128 + 512, planes=256, stride=1)
        self.CBAM2_4 = BasicBlock(inplanes=2 * 64 + 256, planes=128, stride=1)
        self.CBAM2_5 = BasicBlock(inplanes=2 * 64 + 128, planes=64, stride=1)

        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.relu = nn.ReLU()

        # SFN
        self.FDE = FDE()
        self.finalconv1 = nn.Conv2d(128, 64, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.finalconv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                                    padding=1, bias=False)
        self.finalconv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1,
                                    padding=1, bias=False)

        self.conv1d_SCD = nn.Conv2d(64, SCD_class_num, kernel_size=1, stride=1,
                                    padding=0, bias=False)

        # LCM predict
        self.LCM_conv1 = nn.Conv2d(64, 32, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.LCM_conv2 = nn.Conv2d(64, 32, kernel_size=3, stride=1,
                                   padding=1, bias=False)
        self.conv1d_LCM1 = nn.Conv2d(32, LCM_class_num, kernel_size=1, stride=1,
                                     padding=0, bias=False)
        self.conv1d_LCM2 = nn.Conv2d(32, LCM_class_num, kernel_size=1, stride=1,
                                     padding=0, bias=False)

    def forward(self, input1, input2):
        # encoding
        [feat1_1, feat1_2, feat1_3, feat1_4, feat1_5] = self.resnet(input1)
        [feat2_1, feat2_2, feat2_3, feat2_4, feat2_5] = self.resnet(input2)

        # decoding
        cbam1_1 = self.CBAM1_1(torch.cat([feat1_5, feat2_5]))
        cbam1_2 = self.CBAM1_2(torch.cat([self.up(cbam1_1), feat1_4, feat2_4], 1))
        cbam1_3 = self.CBAM1_3(torch.cat([self.up(cbam1_2), feat1_3, feat2_3], 1))
        cbam1_4 = self.CBAM1_4(torch.cat([self.up(cbam1_3), feat1_2, feat2_2], 1))
        cbam1_5 = self.CBAM1_5(torch.cat([self.up(cbam1_4), feat1_1, feat2_1], 1))

        cbam2_1 = self.CBAM2_1(torch.cat([feat1_5, feat2_5]))
        cbam2_2 = self.CBAM2_2(torch.cat([self.up(cbam2_1), feat1_4, feat2_4], 1))
        cbam2_3 = self.CBAM2_3(torch.cat([self.up(cbam2_2), feat1_3, feat2_3], 1))
        cbam2_4 = self.CBAM2_4(torch.cat([self.up(cbam2_3), feat1_2, feat2_2], 1))
        cbam2_5 = self.CBAM2_5(torch.cat([self.up(cbam2_4), feat1_1, feat2_1], 1))

        # LCM predictions
        output1 = self.conv1d_LCM1(self.relu(self.LCM_conv1(self.up(cbam1_5))))
        output2 = self.conv1d_LCM2(self.relu(self.LCM_conv2(self.up(cbam2_5))))

        # SCD output
        output = self.FDE(cbam1_5, cbam2_5)
        output = self.relu(self.finalconv1(output))
        output = self.relu(self.finalconv2(output))
        output = self.relu(self.finalconv3(self.up(output)))

        output = self.conv1d_SCD(output)

        return output, output1, output2

