import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from lib.models.common import Conv, SPP, Bottleneck, BottleneckCSP, Focus, Concat, Detect, SharpenConv, Conv_decoder, C2f, C3, C3TR
from lib.utils import initialize_weights
import cv2
from torchvision import transforms

affine_par = True


class ResnetBlock(nn.Module):
    """Define a Resnet block"""

    def __init__(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Initialize the Resnet block

        A resnet block is a conv block with skip connections
        We construct a conv block with build_conv_block function,
        and implement skip connections in <forward> function.
        Original Resnet paper: https://arxiv.org/pdf/1512.03385.pdf
        """
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        """Construct a convolutional block.

        Parameters:
            dim (int)           -- the number of channels in the conv layer.
            padding_type (str)  -- the name of padding layer: reflect | replicate | zero
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers.
            use_bias (bool)     -- if the conv layer uses bias or not

        Returns a conv block (with a conv layer, a normalization layer, and a non-linearity layer (ReLU))
        """
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim), nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        """Forward function (with skip connections)"""
        out = x + self.conv_block(x)  # add skip connections
        return out


class ResnetGenerator(nn.Module):
    """Resnet-based generator that consists of Resnet blocks between a few downsampling/upsampling operations.

    We adapt Torch code and idea from Justin Johnson's neural style transfer project(https://github.com/jcjohnson/fast-neural-style)
    """

    def __init__(self, input_nc, output_nc, ngf=64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=6, padding_type='reflect'):
        """Construct a Resnet-based generator

        Parameters:
            input_nc (int)      -- the number of channels in input images
            output_nc (int)     -- the number of channels in output images
            ngf (int)           -- the number of filters in the last conv layer
            norm_layer          -- normalization layer
            use_dropout (bool)  -- if use dropout layers
            n_blocks (int)      -- the number of ResNet blocks
            padding_type (str)  -- the name of padding layer in conv layers: reflect | replicate | zero
        """
        assert(n_blocks >= 0)
        super(ResnetGenerator, self).__init__()
        if type(norm_layer) == functools.partial:
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d

        model = [nn.ReflectionPad2d(3),
                 nn.Conv2d(input_nc, ngf, kernel_size=7, padding=0, bias=use_bias),
                 norm_layer(ngf),
                 nn.ReLU(True)]

        n_downsampling = 2
        for i in range(n_downsampling):  # add downsampling layers
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                      norm_layer(ngf * mult * 2),
                      nn.ReLU(True)]

        mult = 2 ** n_downsampling
        for i in range(n_blocks):       # add ResNet blocks

            model += [ResnetBlock(ngf * mult, padding_type=padding_type, norm_layer=norm_layer, use_dropout=use_dropout, use_bias=use_bias)]

        for i in range(n_downsampling):  # add upsampling layers
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2),
                                         kernel_size=3, stride=2,
                                         padding=1, output_padding=1,
                                         bias=use_bias),
                      norm_layer(int(ngf * mult / 2)),
                      nn.ReLU(True)]
        model += [nn.ReflectionPad2d(3)]
        model += [nn.Conv2d(ngf, output_nc, kernel_size=7, padding=0)]
        model += [nn.Tanh()]

        self.model = nn.Sequential(*model)

    def forward(self, input):
        """Standard forward"""
        return self.model(input)


def LightNet():
    model = ResnetGenerator(3, 3, 64, norm_layer=nn.BatchNorm2d, use_dropout=False, n_blocks=3)
    return model


class L_exp_z(nn.Module):
    def __init__(self, patch_size):
        super(L_exp_z, self).__init__()
        self.pool = nn.AvgPool2d(patch_size)

    def forward(self, x, mean_val):
        x = torch.mean(x, 1, keepdim=True)
        mean = self.pool(x)
        d = torch.mean(torch.pow(mean - torch.FloatTensor([mean_val]).cuda(), 2))
        return d


class L_TV(nn.Module):
    def __init__(self, TVLoss_weight=1):
        super(L_TV, self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self, x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = (x.size()[2] - 1) * x.size()[3]
        count_w = x.size()[2] * (x.size()[3] - 1)
        h_tv = torch.pow((x[:, :, 1:, :] - x[:, :, :h_x - 1, :]), 2).sum()
        w_tv = torch.pow((x[:, :, :, 1:] - x[:, :, :, :w_x - 1]), 2).sum()
        return self.TVLoss_weight * 2 * (h_tv / count_h + w_tv / count_w) / batch_size


class SSIM(nn.Module):
    """Layer to compute the SSIM loss between a pair of images
    """
    def __init__(self):
        super(SSIM, self).__init__()
        self.mu_x_pool   = nn.AvgPool2d(3, 1)
        self.mu_y_pool   = nn.AvgPool2d(3, 1)
        self.sig_x_pool  = nn.AvgPool2d(3, 1)
        self.sig_y_pool  = nn.AvgPool2d(3, 1)
        self.sig_xy_pool = nn.AvgPool2d(3, 1)

        self.refl = nn.ReflectionPad2d(1)

        self.C1 = 0.01 ** 2
        self.C2 = 0.03 ** 2

    def forward(self, x, y):
        x = self.refl(x)
        y = self.refl(y)

        mu_x = self.mu_x_pool(x)
        mu_y = self.mu_y_pool(y)

        sigma_x  = self.sig_x_pool(x ** 2) - mu_x ** 2
        sigma_y  = self.sig_y_pool(y ** 2) - mu_y ** 2
        sigma_xy = self.sig_xy_pool(x * y) - mu_x * mu_y

        SSIM_n = (2 * mu_x * mu_y + self.C1) * (2 * sigma_xy + self.C2)
        SSIM_d = (mu_x ** 2 + mu_y ** 2 + self.C1) * (sigma_x + sigma_y + self.C2)

        return torch.clamp((1 - SSIM_n / SSIM_d) / 2, 0, 1)


def edge(img, mode='sobel'):
    add_x_total = torch.zeros(img.shape)

    for i in range(img.shape[0]):
        x = img[i, :, :, :].squeeze(0).cpu().numpy().transpose(1, 2, 0)
        x = x * 255
        if mode=='sobel':
            x_x = cv2.Sobel(x, cv2.CV_64F, 1, 0)
            x_y = cv2.Sobel(x, cv2.CV_64F, 0, 1)
            add_x = cv2.addWeighted(x_x, 0.5, x_y, 0.5, 0)
        elif mode=='scharr':
            x_x = cv2.Scharr(x, cv2.CV_16S, 1, 0)  # X 方向
            x_y = cv2.Scharr(x, cv2.CV_16S, 0, 1)  # Y 方向
            absX = cv2.convertScaleAbs(x_x)
            absY = cv2.convertScaleAbs(x_y)
            add_x = cv2.addWeighted(absX, 0.5, absY, 0.5, 0)
        add_x = transforms.ToTensor()(add_x).unsqueeze(0)
        add_x_total[i, :, :, :] = add_x

    return add_x_total


class enhanceBranch(nn.Module):
    '''relight module: enhance'''
    def __init__(self, ch_in, ch_out=64):
        super().__init__()
        group= 2
        ch_group= ch_out// group
        self.stem_1= Conv(ch_in, ch_group, 3, 2)   # forward
        self.stem_2= Conv(ch_in, ch_group, 3, 2)   # adaptive spatial attention
        self.forward1= nn.Sequential(
            Conv(ch_group, ch_group, 3, 1),
            BottleneckCSP(ch_group, ch_group, 1)
        )
        # spatial attention: maxpool and avgpool
        self.maxpool= nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.avgpool= nn.AvgPool2d(kernel_size=3, stride=2, padding=1)
        self.upConv= nn.Sequential(
            nn.Upsample(None, 2, 'nearest'),
            Conv(ch_out, ch_group, 3, 1)
        )
        self.x1outConv= BottleneckCSP(ch_out, ch_out, 1)

        # unet structure
        self.down_a= nn.Sequential(
            Conv(ch_out, ch_out*2, 3, 2),
            BottleneckCSP(ch_out*2, ch_out*2, 1)
        )
        self.down_b= nn.Sequential(
            Conv(ch_out*2, ch_out*4, 3, 2),
            BottleneckCSP(ch_out*4, ch_out*4, 1)
        )

        self.up_b= nn.Sequential(
            nn.Upsample(None, 2, 'nearest'),
            Conv(ch_out*4, ch_out*2, 3, 1),
            BottleneckCSP(ch_out*2, ch_out*2, 1)
        )
        self.up_a= nn.Sequential(
            Conv(ch_out*4, ch_out, 3, 1),
            BottleneckCSP(ch_out, ch_out, 1)
        )
        self.up_out = nn.Upsample(None, 2, 'nearest')
        self.x2outConv= nn.Sequential(
            Conv(ch_out*2, ch_out, 3, 1),
            BottleneckCSP(ch_out, ch_out, 1),
        )


    def forward(self, x):
        # spatial attention
        s1= self.stem_1(x)   # stream1
        s2= self.stem_2(x)   # stream2

        s1= self.forward1(s1)
        s2_ave= self.avgpool(s2)
        s2_max= self.maxpool(s2)   # ave+max
        s2_pool= torch.cat([s2_ave, s2_max], dim=1)
        s2= self.upConv(s2_pool)+ s2
        x1out= self.x1outConv(torch.cat([s1, s2], dim=1))

        # unet
        xdowna= self.down_a(x1out)
        xdownb= self.down_b(xdowna)
        xupa= self.up_b(xdownb)

        xupa= self.up_a(torch.cat([xupa, xdowna], dim=1))
        xup= self.up_out(xupa)
        x2out= self.x2outConv(torch.cat([xup, x1out], dim=1))+ x1out

        return x2out


class detailBranch(nn.Module):
    '''relight module: detail'''
    def __init__(self, ch_in=3, ch_out=64, mode='sobel'):
        super().__init__()
        self.mode= mode
        self.edgedown1 = Conv(ch_in, ch_out, 3, 2)
        self.edgedown2 = Conv(ch_out, ch_out*2, 3, 2)
        self.edgedown3 = Conv(ch_out*2, ch_out*4, 3, 2)

        self.down1= Conv(ch_in, ch_out, 3, 2)
        self.down2= Conv(ch_out*2, ch_out*2, 3, 2)
        self.down3= Conv(ch_out*4, ch_out*4, 3, 2)

        self.up3= nn.Sequential(
            nn.Upsample(None, 2, 'nearest'),
            BottleneckCSP(ch_out*8, ch_out*4, 1, False)
        )
        self.up2= nn.Sequential(
            nn.Upsample(None, 2, 'nearest'),
            BottleneckCSP(ch_out*4, ch_out, 1, False)
        )

    def forward(self, x):
        x_edge= edge(x, mode=self.mode).to(x.device)
        x_edge_down1= self.edgedown1(x_edge)
        x_edge_down2= self.edgedown2(x_edge_down1)
        x_edge_down3= self.edgedown3(x_edge_down2)

        x_down1= self.down1(x)
        x_down2= self.down2(torch.cat([x_edge_down1, x_down1], dim=1))
        x_down3= self.down3(torch.cat([x_edge_down2, x_down2], dim=1))

        x_up3= self.up3(torch.cat([x_edge_down3, x_down3], dim=1))
        x_out= self.up2(x_up3)+ x_down1
        return x_out


class relight(nn.Module):
    '''relight module'''
    def __init__(self, ch_in, ch_inner, mode='sobel'):
        super().__init__()
        self.mode= mode
        self.enhance= enhanceBranch(ch_in, ch_inner)
        self.detail = detailBranch(ch_in, ch_inner, mode)

        self.up= nn.Sequential(
            nn.Upsample(None, 2, 'nearest'),
            BottleneckCSP(ch_inner*2, ch_in, 1, False)
            #BottleneckCSP(ch_inner, ch_in, 1, False)
        )
        initialize_weights(self)

    def forward(self, x):
        x_en= self.enhance(x)
        x_de= self.detail(x)
        #out = self.up(x_de)+ x
        out= self.up(torch.cat([x_en, x_de], dim=1))+ x
        return out


if __name__ == '__main__':
    #x= torch.rand([2,3,256,256])
    #model= relight(3, 64, 'sobel')
    #print(model(x).shape)

    img= cv2.imread(r'D:\zezeze\MultitaskDAN\test.jpg')
    x= torch.tensor(img).unsqueeze(0).permute(0, 3, 1, 2)
    out = edge(x, mode='canny')
    out= out.permute(0, 2, 3, 1).numpy()
    cv2.imshow('img', out[0])
    cv2.waitKey()
