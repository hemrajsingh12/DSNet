import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class SConv2d(nn.Module):
    def __init__(self, inplanes, planes, kernel_size, stride=1, padding=0, dilation=1):
        super(SConv2d, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.Conv2d(inplanes, planes,
                      kernel_size=kernel_size, stride=stride,
                      padding=padding, dilation=dilation, bias=False),
            nn.BatchNorm2d(planes)
        )
        
    def forward(self, x):
        x = self.conv_bn(x)
        return x


class BCFM(nn.Module):
	"""Bi-Directional Cross Fusion Module """
	def __init__(self, inplanes, planes, rates=[1, 6, 12, 18]):
		super(BCFM, self).__init__()

		self.dcfm0 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=1,
			stride=1, padding=0, dilation=1, bias=False),
			nn.BatchNorm2d(planes))
		self.dcfm1 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
			stride=1, padding=rates[1], dilation=rates[1], bias=False),
			nn.BatchNorm2d(planes))        
		self.dcfm2 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
			stride=1, padding=rates[2], dilation=rates[2], bias=False),
			nn.BatchNorm2d(planes))
		self.dcfm3 = nn.Sequential(nn.Conv2d(inplanes, planes, kernel_size=3,
			stride=1, padding=rates[3], dilation=rates[3], bias=False),
			nn.BatchNorm2d(planes))

		self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False))

		self.reduce = nn.Sequential(
			nn.Conv2d(planes*5, planes, kernel_size=1, bias=False),
			nn.BatchNorm2d(planes))

    self.sconv = nn.Sequential(
      nn.Conv2d(planes, planes, kernel_size=1,bias=False),
      nn.ReLU())

	def forward(self, x, y):
		x0 = self.dcfm0(x)
		x1 = self.dcfm1(x)
		x2 = self.dcfm2(x)
		x3 = self.dcfm3(x)
		x4 = self.global_avg_pool(x)
		x4 = F.upsample(x4, x3.size()[2:], mode='bilinear', align_corners=True)
		x = torch.cat((x0, x1, x2, x3, x4), dim=1)
		x = self.reduce(x)
        y0 = self.dcfm0(y)
		y1 = self.dcfm1(y)
		y2 = self.dcfm2(y)
		y3 = self.dcfm3(y)
        y4 = self.global_avg_pool(y)
		y4 = F.upsample(y4, y3.size()[2:], mode='bilinear', align_corners=True)
		y = torch.cat((y0, y1, y2, y3, y4), dim=1)
		y = self.reduce(y)
        x = self.sconv(x) * self.sconv(y)
        y = self.sconv(y) * self.conv(x)
        x = self.global_avg_pool(x)
		x = F.upsample(x, x3.size()[2:], mode='bilinear', align_corners=True)
        y = self.global_avg_pool(x)
		y = F.upsample(y, y3.size()[2:], mode='bilinear', align_corners=True)
		return x, y

class DAFM_module(nn.Module):
	def __init__(self, inplanes, planes, rate):
		super(DAFM_module, self).__init__()
		if rate == 1:
			kernel_size = 1
			padding = 0
		else:
			kernel_size = 3
			padding = rate
		self.atrous_convolution = nn.Conv2d(inplanes, planes, kernel_size=kernel_size,
			stride=1, padding=padding, dilation=rate, bias=False)
		self.bn = nn.BatchNorm2d(planes)
		self.relu = nn.PReLU()

		self.__init_weight()

	def forward(self, x):
		x = self.atrous_convolution(x)
		x = self.bn(x)

		return self.relu(x)

	def __init_weight(self):
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				# n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
				# m.weight.data.normal_(0, math.sqrt(2. / n))
				torch.nn.init.kaiming_normal_(m.weight)
			elif isinstance(m, nn.BatchNorm2d):
				m.weight.data.fill_(1)
				m.bias.data.zero_()

class SPM(nn.Module):
    """
        Saliency Prediction module (SPM)
    """
    def __init__(self,inplanes, planes, **kwargs):
        super(SPM, self).__init__()
        inter_channels = int(inplanes / 4)
        self.conv1 = nn.Conv2d(inplanes, inter_channels, 1, **kwargs)
        self.conv2 = nn.Conv2d(inplanes, inter_channels, 1, **kwargs)
        self.conv3 = nn.Conv2d(inplanes, inter_channels, 1, **kwargs)
        self.conv4 = nn.Conv2d(inplanes, inter_channels, 1, **kwargs)
        self.out = nn.Conv2d(inplanes * 2, planes, 1)

    def pool(self, x, size):
        avgpool = nn.AdaptiveAvgPool2d(size)
        return avgpool(x)

    def upsample(self, x, size):
        return F.interpolate(x, size, mode='bilinear', align_corners=True)

    def forward(self, x, y):
        size = x.size()[2:]

        feat1 = self.upsample(self.conv1(self.pool(x, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(x, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(x, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(x, 6)), size)
        x = torch.cat([x, feat1, feat2, feat3, feat4], dim=1)
        x = self.out(x)
        size = y.size()[2:]

        feat1 = self.upsample(self.conv1(self.pool(y, 1)), size)
        feat2 = self.upsample(self.conv2(self.pool(y, 2)), size)
        feat3 = self.upsample(self.conv3(self.pool(y, 3)), size)
        feat4 = self.upsample(self.conv4(self.pool(y, 6)), size)
        y = torch.cat([y, feat1, feat2, feat3, feat4], dim=1)
        y = self.out(y)

        return x

class DAFM(nn.Module):
	def __init__(self, inplanes, planes, rates):
		super(DAFM, self).__init__()

		self.aspp1 = DAFM_module(inplanes, planes, rate=rates[0])
		self.aspp2 = DAFM_module(inplanes, planes, rate=rates[1])
		self.aspp3 = DAFM_module(inplanes, planes, rate=rates[2])
		self.aspp4 = DAFM_module(inplanes, planes, rate=rates[3])

		self.relu = nn.ReLU()

		self.global_avg_pool = nn.Sequential(nn.AdaptiveAvgPool2d((1, 1)),
			nn.Conv2d(inplanes, planes, 1, stride=1, bias=False),
			nn.BatchNorm2d(planes),
			nn.PReLU()
		)

		self.conv1 = nn.Conv2d(planes*5, planes, 1, bias=False)
		self.bn1 = nn.BatchNorm2d(planes)

	def forward(self, x, y):
		x1 = self.aspp1(x)
		x2 = self.aspp2(x)
		x3 = self.aspp3(x)
		x4 = self.aspp4(x)
		x5 = self.global_avg_pool(x)
		x5 = F.upsample(x5, size=x4.size()[2:], mode='bilinear', align_corners=True)

		x = torch.cat((x1, x2, x3, x4, x5), dim=1)

		x = self.conv1(x)
		x = self.bn1(x)
		x = self.relu(x)

        y1 = self.aspp1(y)
		y2 = self.aspp2(y)
		y3 = self.aspp3(y)
		y4 = self.aspp4(y)
		y5 = self.global_avg_pool(y)
		y5 = F.upsample(y5, size=x4.size()[2:], mode='bilinear', align_corners=True)

		y = torch.cat((y1, y2, y3, y, y5), dim=1)

		y = self.conv1(y)
		y = self.bn1(y)
		y = self.relu(y)
        fuse = x + y
		
		return fuse #x 
