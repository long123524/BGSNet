import timm
import torch.nn.functional as F
from lib.pvtv2 import pvt_v2_b2
import torch
import torch.nn as nn
import torch.nn.functional as F
import warnings
from torchvision import models


class BasicConv2d(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1):
        super(BasicConv2d, self).__init__()

        self.conv = nn.Conv2d(in_planes, out_planes,
                              kernel_size=kernel_size, stride=stride,
                              padding=padding, dilation=dilation, bias=False)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

######
class CBAM(nn.Module):
    def __init__(self, channel, reduction=16, spatial_kernel=7):
        super(CBAM, self).__init__()
        # channel attention 压缩H,W为1
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # shared MLP
        self.mlp = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, 1, bias=False)
        )
        # spatial attention
        self.conv = nn.Conv2d(2, 1, kernel_size=spatial_kernel,
                              padding=spatial_kernel // 2, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        max_out = self.mlp(self.max_pool(x))
        avg_out = self.mlp(self.avg_pool(x))
        channel_out = self.sigmoid(max_out + avg_out)
        x = channel_out * x
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        avg_out = torch.mean(x, dim=1, keepdim=True)
        spatial_out = self.sigmoid(self.conv(torch.cat([max_out, avg_out], dim=1)))
        x = spatial_out * x
        return x

class Residual(nn.Module):
    def __init__(self, input_dim, output_dim, stride=1, padding=1):
        super(Residual, self).__init__()

        self.conv_block = nn.Sequential(
            nn.BatchNorm2d(input_dim),
            nn.ReLU(),
            nn.Conv2d(
                input_dim, output_dim, kernel_size=3, stride=stride, padding=padding
            ),
            nn.BatchNorm2d(output_dim),
            nn.ReLU(),
            nn.Conv2d(output_dim, output_dim, kernel_size=3, padding=1),
        )
        self.conv_skip = nn.Sequential(
            nn.Conv2d(input_dim, output_dim, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(output_dim),
        )

    def forward(self, x):

        return self.conv_block(x) + self.conv_skip(x)


## boundary-contextual guided module (BCG)
class BCG(nn.Module):
    def __init__(self, in_dim):
        super(BCG, self).__init__()
        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x, boundary):
        m_batchsize, C, height, width = x.size()
        q = self.query_conv1(x).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k = self.key_conv1(boundary).view(m_batchsize, -1, width * height)
        v = self.value_conv1(x).view(m_batchsize, -1, width * height)
        energy1 = torch.bmm(q, k)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)
        out1 = x + out1

        return out1


class CotSR(nn.Module):
    #
    def __init__(self, in_dim):
        super(CotSR, self).__init__()
        self.chanel_in = in_dim

        self.query_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv1 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.query_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.key_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim // 4, kernel_size=1)
        self.value_conv2 = nn.Conv2d(in_channels=in_dim, out_channels=in_dim, kernel_size=1)

        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x1, x2):
        ''' inputs :
                x1 : input feature maps( B X C X H X W)
                x2 : input feature maps( B X C X H X W)
            returns :
                out : attention value + input feature
                attention: B X (HxW) X (HxW) '''
        m_batchsize, C, height, width = x1.size()

        q1 = self.query_conv1(x1).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k1 = self.key_conv1(x1).view(m_batchsize, -1, width * height)
        v1 = self.value_conv1(x1).view(m_batchsize, -1, width * height)

        q2 = self.query_conv2(x2).view(m_batchsize, -1, width * height).permute(0, 2, 1)
        k2 = self.key_conv2(x2).view(m_batchsize, -1, width * height)
        v2 = self.value_conv2(x2).view(m_batchsize, -1, width * height)

        energy1 = torch.bmm(q1, k2)
        attention1 = self.softmax(energy1)
        out1 = torch.bmm(v2, attention1.permute(0, 2, 1))
        out1 = out1.view(m_batchsize, C, height, width)

        energy2 = torch.bmm(q2, k1)
        attention2 = self.softmax(energy2)
        out2 = torch.bmm(v1, attention2.permute(0, 2, 1))
        out2 = out2.view(m_batchsize, C, height, width)

        out1 = x1 + self.gamma1 * out1
        out2 = x2 + self.gamma2 * out2

        return out1, out2


##
class MSD(nn.Module):
    '''
    Three-layer CNN for encoding map info.
    '''
    def __init__(
        self,
        input_channels,
        out_channels,
    ):
        super(MSD, self).__init__()
        self.out_channels = out_channels
        self.input_channels= input_channels
        self.layer1 = BasicConv2d(input_channels,out_channels,1)
        self.layer2 = BasicConv2d(out_channels,out_channels,3, 1, 1, 1)
        self.layer3 = BasicConv2d(out_channels, out_channels, 5, 1, 2, 1)

    def forward(self, edge):
        x = self.layer1(edge)
        x1 = edge + x
        x2 = self.layer2(x1)
        x3 = x2 + x1
        x4 = self.layer3(x3)
        x_total = x4 + x3

        return x_total


def conv1x1(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class FCN(nn.Module):
    def __init__(self, in_channels=3, pretrained=True):
        super(FCN, self).__init__()
        resnet = models.resnet34(pretrained)
        newconv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
        newconv1.weight.data[:, 0:3, :, :].copy_(resnet.conv1.weight.data[:, 0:3, :, :])
        if in_channels > 3:
            newconv1.weight.data[:, 3:in_channels, :, :].copy_(resnet.conv1.weight.data[:, 0:in_channels - 3, :, :])

        self.layer0 = nn.Sequential(newconv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        for n, m in self.layer3.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)
        for n, m in self.layer4.named_modules():
            if 'conv1' in n or 'downsample.0' in n:
                m.stride = (1, 1)

    def _make_layer(self, block, inplanes, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or inplanes != planes:
            downsample = nn.Sequential(
                conv1x1(inplanes, planes, stride),
                nn.BatchNorm2d(planes))

        layers = []
        layers.append(block(inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)


##
class BGSNet(nn.Module):
    def __init__(self, channel=32,num_classes=7, pretrained_path=r'D:\SCD\pretrained\pvt_v2_b2.pth', drop_rate = 0.4):
        super(BGSNet, self).__init__()
        self.drop = nn.Dropout2d(drop_rate)
        self.backbone = pvt_v2_b2()  # [64, 128, 320, 512]
        save_model = torch.load(pretrained_path)
        model_dict = self.backbone.state_dict()
        state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
        model_dict.update(state_dict)
        self.backbone.load_state_dict(model_dict)
        self.channel = channel
        self.num_classes = num_classes
        self.Translayer1_1 = BasicConv2d(64, channel, 1)
        self.Translayer2_1 = BasicConv2d(128, channel, 1)
        self.Translayer3_1 = BasicConv2d(320, channel*2, 1)
        self.Translayer4_1 = BasicConv2d(512, channel*2, 1)
        self.Translayer5_1 = BasicConv2d(192, channel, 1)

        self.up1 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up2 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.up3 = nn.Upsample(scale_factor=4, mode='bilinear', align_corners=True)
        self.up4 = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        #
        self.out_feature = nn.Conv2d(64, num_classes, 1)
        self.out_feature2 = nn.Conv2d(64, num_classes, 1)
        self.out_feature1 = nn.Conv2d(64, 1, 1)
        self.C1 = CBAM(128)
        self.C0 = CBAM(64)

        self.ms = MSD(64, 64)
        self.cot = CotSR(64)

        self.bcg = BCG(64)
        self.cov16 = BasicConv2d(64,64,3, 1, 1, 1)
    def base_forward(self, x):
        pvt = self.backbone(x)
        x1 = pvt[0]
        x1 = self.drop(x1)
        x2 = pvt[1]
        x2 = self.drop(x2)
        x3 = pvt[2]
        x3 = self.drop(x3)
        x4 = pvt[3]
        x4 = self.drop(x4)
        ##level1
        x1 = self.Translayer1_1(x1)
        ###level2
        x2_2 = self.Translayer2_1(x2)
        x2_3 = self.up1(x2_2)
        ##
        x_low = x1 + x2_3
        x_low = self.C0(x_low)
        ###level3
        x3_2 = self.Translayer3_1(x3)

        ###level4
        x4_2 = self.Translayer4_1(x4)
        x4_3 = self.up2(x4_2)
        ##
        x_high = x3_2 + x4_3
        x_high = self.C1(x_high)
        x_total= torch.cat((x_low, self.up3(x_high)), 1)
        x_fuse = self.Translayer5_1(x_total)

        return x_fuse

    def forward(self, t1, t2):

        t1_out = self.base_forward(t1)
        t2_out = self.base_forward(t2)
        t1_out, t2_out = self.cot(t1_out, t2_out)

        changes = torch.abs(t2_out-t1_out)
        b_changes = torch.abs(t2_out-t1_out)

        boundary = self.ms(b_changes) #
        out_edge = self.out_feature3(boundary)

        change_areas = self.cov16(changes)
        out_c = self.bcg(change_areas, boundary)
        out_cs = self.out_feature1(out_c)

        ###
        output1 = self.out_feature(t1_out)
        output2 = self.out_feature2(t2_out)

        ##output
        prediction1_1 = F.interpolate(out_cs, scale_factor=4, mode='bilinear')
        prediction1_2 = F.interpolate(output1, scale_factor=4, mode='bilinear')
        prediction1_3 = F.interpolate(output2, scale_factor=4, mode='bilinear')
        predict_edge = F.interpolate(out_edge, scale_factor=4, mode='bilinear')

        return prediction1_1, prediction1_2, prediction1_3, predict_edge







