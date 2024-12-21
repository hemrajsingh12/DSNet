import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.ops

def upsample(x, size):
    return F.interpolate(x, size, mode='bilinear', align_corners=True)


class DepthBranch(nn.Module):
    def __init__(self, c1=8, c2=16, c3=32, c4=48, c5=320, **kwargs):
        super(DepthBranch, self).__init__()
        self.bottleneck1 = _make_layer(LinearBottleneck, 1, 16, blocks=1, t=3, stride=2)
        self.bottleneck2 = _make_layer(LinearBottleneck, 16, 24, blocks=3, t=3, stride=2)
        self.bottleneck3 = _make_layer(LinearBottleneck, 24, 32, blocks=7, t=3, stride=2)
        self.bottleneck4 = _make_layer(LinearBottleneck, 32, 96, blocks=3, t=2, stride=2)
        self.bottleneck5 = _make_layer(LinearBottleneck, 96, 320, blocks=1, t=2, stride=1)

        # self.conv_s_d = _ConvBNReLU(320,1,1,1)

        # nn.Sequential(_DSConv(c3, c3 // 4),
        #                           nn.Conv2d(c3 // 4, 1, 1), )

    def forward(self, x):
        size = x.size()[2:]
        feat = []

        x1 = self.bottleneck1(x)
        x2 = self.bottleneck2(x1)
        x3 = self.bottleneck3(x2)
        x4 = self.bottleneck4(x3)
        x5 = self.bottleneck5(x4)
        # s_d = self.conv_s_d(x5)

        feat.append(x1)
        feat.append(x2)
        feat.append(x3)
        feat.append(x4)
        feat.append(x5)
        return feat
#............................. Channel Attention (CA) ........
class CA(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(CA, self).__init__()

        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc1 = nn.Conv2d(in_planes, in_planes // 16, 1, bias=False)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Conv2d(in_planes // 16, in_planes, 1, bias=False)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out = self.fc2(self.relu1(self.fc1(self.max_pool(x))))
        out = max_out
        return self.sigmoid(out)


class SA(nn.Module):
    def __init__(self, kernel_size=7):
        super(SA, self).__init__()

        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1

        self.conv1 = nn.Conv2d(1, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = max_out
        x = self.conv1(x)
        return self.sigmoid(x)
#............................. DARP............
class DARP(nn.Module):
    def __init__(self, in_channels):
        super(DARP, self).__init__()
        self.d1 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d2 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d3 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d4 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d5 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.d6 = nn.Sequential(DepthBranch(in_channels), nn.BatchNorm2d(in_channels),nn.ReLU())
        self.sa = SA()
        self.ca = CA()
        
    def forward(self, x):
        size = x.size()[2:]
        feat = []

        x1 = self.d1(x)
        x11 = self.ca(x1)
        x1 = x1 * x11
        x2 = self.d2(x1)
        x22 = self.ca(x2)
        x2 = x2 * x22
        x3 = self.d3(x2)
        x33 = self.ca(x3)
        x3 = x3 * x33
        x4 = self.d4(x3)
        x44 = self.sa(x4)
        x4 = x4 * x44
        x5 = self.d5(x4)
        x55 = self.sa(x5)
        x5 = x5 * x55
        x6 = self.d6(x5)
        x66 = self.ca(x6)
        x6 = x6 * x66
        #....... 
        feat.append(x1)
        feat.append(x2)
        feat.append(x3)
        feat.append(x4)
        feat.append(x5)
        return feat
#......................... SCNet...............
class SCNet(nn.Module):
    """Separable convolution Network(SCNet)"""

    def __init__(self, in_channels, out_channels, **kwargs):
        super(SCNet, self).__init__()
        inter_channels = int(in_channels / 4)
        self.conv1 = DSConv(in_channels, inter_channels, 1, **kwargs)
        self.conv2 = DSConv(in_channels, inter_channels, 1, **kwargs)
        self.conv3 = DSConv(in_channels, inter_channels, 1, **kwargs)
        self.conv4 = DSConv(in_channels, inter_channels, 1, **kwargs)
        self.out = DSConv(in_channels * 2, out_channels, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def forward(self, x):
        size = x.size()[2:]
        feat1 = upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        return x

class ConvBNReLU(nn.Module):
    """Conv-BN-ReLU"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(ConvBNReLU, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

class ConvBNSig(nn.Module):
    """Conv-BN-Sigmoid"""

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,dilation=1, **kwargs):
        super(ConvBNSig, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding,dilation=dilation ,bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.conv(x)


class DSConv(nn.Module):
    """Depthwise Separable Convolutions"""

    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DSConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, dw_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(dw_channels),
            nn.ReLU(True),
            nn.Conv2d(dw_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)

def _make_layer( block, inplanes, planes, blocks, t=6, stride=1):
    layers = []
    layers.append(block(inplanes, planes, t, stride))
    for i in range(1, blocks):
        layers.append(block(planes, planes, t, 1))
    return nn.Sequential(*layers)

class DWConv(nn.Module):
    def __init__(self, dw_channels, out_channels, stride=1, **kwargs):
        super(_DWConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(dw_channels, out_channels, 3, stride, 1, groups=dw_channels, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(True)
        )

    def forward(self, x):
        return self.conv(x)


class LinearBottleneck(nn.Module):
    """LinearBottleneck used in MobileNetV2"""

    def __init__(self, in_channels, out_channels, t=6, stride=2, **kwargs):
        super(LinearBottleneck, self).__init__()
        self.use_shortcut = stride == 1 and in_channels == out_channels
        self.block = nn.Sequential(
            # pw
            ConvBNReLU(in_channels, in_channels * t, 1),
            ConvBNSig(in_channels, in_channels * t, stride),
            DSConv(in_channels, in_channels * t, stride)

            # dw
            DWConv(in_channels * t, in_channels * t, stride),
            # pw-linear
            nn.Conv2d(in_channels * t, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        )

    def forward(self, x):
        out = self.block(x)
        if self.use_shortcut:
            out = x + out
        return out

class DeCNet(nn.Module):
    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=3,
                 stride=1,
                 padding=1,
                 bias=False):

        super(DeCNet, self).__init__()
        
        assert type(kernel_size) == tuple or type(kernel_size) == int

        kernel_size = kernel_size if type(kernel_size) == tuple else (kernel_size, kernel_size)
        self.stride = stride if type(stride) == tuple else (stride, stride)
        self.padding = padding
        
        self.offset_conv = nn.Conv2d(in_channels, 
                                     2 * kernel_size * kernel_size * out_channels,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        
        self.modulator_conv = nn.Conv2d(2 * kernel_size * kernel_size * out_channels, 
                                     1 * kernel_size * kernel_size * out_channels,
                                     kernel_size=kernel_size, 
                                     stride=stride,
                                     padding=self.padding, 
                                     bias=True)
        
        self.regular_conv = nn.Conv2d(in_channels=1 * kernel_size * kernel_size * out_channels,
                                      out_channels=out_channels,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      padding=self.padding,
                                      bias=bias)

    def forward(self, x):
        x = self.offset_conv(x)
        x_mod = x + self.modulator_conv(x)
        x_reg = x_mod + self.regular_conv(x_mod)
        return x


class DSNet(nn.Module):
    def __init__(self, kernel_size=3, padding=1):
        super(DSNet, self).__init__()
        self.conv1_1 = DeCNet(3, 32, kernel_size=kernel_size, padding=padding)
        self.conv1_2 = SCNet(32, 32, kernel_size=kernel_size, padding=padding)
        self.drap1 = DARP(32)
        self.maxpool1 = nn.MaxPool2d(2)

        self.conv2_1 = DeCNet(32, 64, kernel_size=kernel_size, padding=padding)
        self.conv2_2 = SCNet(64, 64, kernel_size=kernel_size, padding=padding)
        self.drap2 = DARP(64)
        self.maxpool2 = nn.MaxPool2d(2)

        self.conv3_1 = DeCNet(64, 128, kernel_size=kernel_size, padding=padding)
        self.conv3_2 = SCNet(128, 128, kernel_size=kernel_size, padding=padding)
        self.drap3 = DARP(128)
        self.maxpool3 = nn.MaxPool2d(2)

        self.conv4_1 = DeCNet(128, 256, kernel_size=kernel_size, padding=padding)
        self.conv4_2 = SCNet(256, 256, kernel_size=kernel_size, padding=padding)
        self.drap4 = DARP(256)
        self.maxpool4 = nn.MaxPool2d(2)

        self.conv5_1 = DeCNet(256, 512, kernel_size=kernel_size, padding=padding)
        self.conv5_2 = SCNet(512, 512, kernel_size=kernel_size, padding=padding)
        self.conv5_t = nn.ConvTranspose2d(512, 256, 2, stride=2)

        self.conv6_1 = SCNet(512, 256, kernel_size=kernel_size, padding=padding)
        self.conv6_2 = DeCNet(256, 256, kernel_size=kernel_size, padding=padding)
        self.conv6_t = nn.ConvTranspose2d(256, 128, 2, stride=2)

        self.conv7_1 = SCNet(256, 128, kernel_size=kernel_size, padding=padding)
        self.conv7_2 = DeCNet(128, 128, kernel_size=kernel_size, padding=padding)
        self.conv7_t = nn.ConvTranspose2d(128, 64, 2, stride=2)

        self.conv8_1 = SCNet(128, 64, kernel_size=kernel_size, padding=padding)
        self.conv8_2 = DeCNet(64, 64, kernel_size=kernel_size, padding=padding)
        self.conv8_t = nn.ConvTranspose2d(64, 32, 2, stride=2)

        self.conv9_1 = SCNet(64, 32, kernel_size=kernel_size, padding=padding)
        self.conv9_2 = DeCNet(32, 32, kernel_size=kernel_size, padding=padding)

        self.conv10 = nn.Conv2d(32, 1, kernel_size=kernel_size, padding=padding)

    def forward(self, x):
        conv1 = F.relu(self.conv1_1(x))
        conv1 = F.relu(self.conv1_2(conv1))
        conv1 = self.drap1(conv1)
        pool1 = self.maxpool1(conv1)

        conv2 = F.relu(self.conv2_1(pool1))
        conv2 = F.relu(self.conv2_2(conv2))
        conv2 = self.drap2(conv2)
        pool2 = self.maxpool2(conv2)

        conv3 = F.relu(self.conv3_1(pool2))
        conv3 = F.relu(self.conv3_2(conv3))
        conv3 = self.drap3(conv3)
        pool3 = self.maxpool3(conv3)

        conv4 = F.relu(self.conv4_1(pool3))
        conv4 = F.relu(self.conv4_2(conv4))
        conv4 = self.drap4(conv4)
        pool4 = self.maxpool4(conv4)

        conv5 = F.relu(self.conv5_1(pool4))
        conv5 = F.relu(self.conv5_2(conv5))

        up6 = torch.cat((self.conv5_t(conv5), conv4), dim=1)
        conv6 = F.relu(self.conv6_1(up6))
        conv6 = F.relu(self.conv6_2(conv6))

        up7 = torch.cat((self.conv6_t(conv6), conv3), dim=1)
        conv7 = F.relu(self.conv7_1(up7))
        conv7 = F.relu(self.conv7_2(conv7))

        up8 = torch.cat((self.conv7_t(conv7), conv2), dim=1)
        conv8 = F.relu(self.conv8_1(up8))
        conv8 = F.relu(self.conv8_2(conv8))

        up9 = torch.cat((self.conv8_t(conv8), conv1), dim=1)
        conv9 = F.relu(self.conv9_1(up9))
        conv9 = F.relu(self.conv9_2(conv9))

        return F.sigmoid(self.conv10(conv9))

    