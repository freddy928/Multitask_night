import torch
import torch.nn as nn
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, Conv_decoder, C2f, C3, C3TR


class multiDiscriminator(nn.Module):
	'''multitask Discriminator'''
	def __init__(self, ch_det=18, ch_inner=64):
		super().__init__()
		self.ch_det= ch_det
		self.ch_da= 2
		self.ch_ll= 2
		self.up32_16= nn.Upsample(None, 2, 'nearest')
		self.detconv32_16= nn.Sequential(
			Conv(ch_det*2, ch_det, 3, 1),
			BottleneckCSP(ch_det, ch_det, 1),
		)
		self.up16_8= nn.Upsample(None, 2, 'nearest')
		self.detconv16_8= nn.Sequential(
			Conv(ch_det * 2, ch_det, 3, 1),
			BottleneckCSP(ch_det, ch_det, 1),
		)
		self.det_inner= Conv(ch_det, ch_inner, 3, 1)
		self.da_inner= nn.Sequential(
			Conv(self.ch_ll, ch_inner//4, 3, 2),
			BottleneckCSP(ch_inner//4, ch_inner//4, 1),
			Conv(ch_inner//4, ch_inner//2, 3, 2),
			BottleneckCSP(ch_inner//2, ch_inner//2, 1),
			Conv(ch_inner//2, ch_inner, 3, 2)
		)
		self.ll_inner= nn.Sequential(
			Conv(self.ch_ll, ch_inner//4, 3, 2),
			BottleneckCSP(ch_inner//4, ch_inner//4, 1),
			Conv(ch_inner//4, ch_inner//2, 3, 2),
			BottleneckCSP(ch_inner//2, ch_inner//2, 1),
			Conv(ch_inner//2, ch_inner, 3, 2)
		)
		self.classifier = nn.Conv2d(ch_inner* 3, 1, kernel_size=3, stride=1, padding=1)

	def forward(self, input):

		det, da, ll= input
		b, ch1, ny, nx, ch2= det[0].shape
		detinput= [None, None, None]
		for i in range(len(det)):
			detinput[i]= det[i].permute(0, 1, 4, 2, 3).reshape(b, self.ch_det, ny//(2**i), nx//(2**i))
		detinput[2]= self.up32_16(detinput[2])
		detinput[1]= self.detconv32_16(torch.cat([detinput[1], detinput[2]], dim=1))
		detinput[1]= self.up16_8(detinput[1])
		detfeat= self.det_inner(self.detconv16_8(torch.cat([detinput[0], detinput[1]], dim=1)))

		dafeat= self.da_inner(da)
		llfeat= self.ll_inner(ll)

		out= self.classifier(torch.cat([detfeat, dafeat, llfeat], dim=1))

		return out


if __name__ == '__main__':
	det= [torch.rand([2, 3, 32, 64, 6]), torch.rand([2, 3, 16, 32, 6]), torch.rand([2, 3, 8, 16, 6])]
	da= torch.rand([2, 2, 256, 512])
	ll= torch.rand([2, 2, 256, 512])
	model= multiDiscriminator(ch_det=18, ch_inner=64)
	print(model(det,da,ll).shape)