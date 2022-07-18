import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from basicsr.models.archs.base_network import BaseNetwork
from basicsr.models.archs.normalization import get_nonspade_norm_layer
import torch

class ResnetBlock(nn.Module):
    def __init__(self, dim, act, kernel_size=3):
        super().__init__()
        self.act = act
        pw = (kernel_size - 1) // 2
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            act,
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size)
        )

    def forward(self, x):
        y = self.conv_block(x)
        out = x + y
        return self.act(out)

class ConvEncoder(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        self.ndf = ndf
        norm_E = "spectralinstance"
        norm_layer = get_nonspade_norm_layer(None, norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        # self.layer2_1 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        # self.layer3_1 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, kw, stride=1, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        # self.layer4_1 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        # self.layer7 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        # self.layer8 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        self.res_0 = ResnetBlock(ndf*8, nn.LeakyReLU(0.2, False))
        self.res_1 = ResnetBlock(ndf*8, nn.LeakyReLU(0.2, False))
        self.res_2 = ResnetBlock(ndf*8, nn.LeakyReLU(0.2, False))
        self.res_3 = ResnetBlock(ndf*8, nn.LeakyReLU(0.2, False))
        self.res_4 = ResnetBlock(ndf*8, nn.LeakyReLU(0.2, False))
        self.fc_0 = nn.Linear(6*6*ndf*8, 4*4*ndf)
        self.fc_1 = nn.Linear(ndf*4*4, ndf*4*4)
        self.so = s0 = 4
        self.layer8_1 = norm_layer(nn.Conv2d(ndf * 2, ndf, kw, stride=1, padding=pw))
        self.out = norm_layer(nn.Conv2d(ndf * 8, ndf * 4, kw, stride=1, padding=0))
        self.down = nn.AvgPool2d(2,2)
        # self.global_avg = nn.AdaptiveAvgPool2d((6,6))
        
        
        self.layer8_2 = nn.Conv2d(ndf, 16, kw, stride=1, padding=pw)
        self.layer9_1 = norm_layer(nn.Conv2d(ndf * 2, ndf, kw, stride=1, padding=pw))
        self.layer9_2 = nn.Conv2d(ndf, 16, kw, stride=1, padding=pw)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.pad_3 = nn.ReflectionPad2d(3)
        self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_7x7 = nn.Conv2d(ndf, ndf, kernel_size=7, padding=0, bias=True)
        # self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x) # 128
        x = self.conv_7x7(self.pad_3(self.actvn(x)))
        x = self.layer2(self.actvn(x)) # 64
        # x = self.layer2_1(self.actvn(x))
        x = self.layer3(self.actvn(x)) # 32
        # x = self.layer3_1(self.actvn(x))
        x = self.layer4(self.actvn(x)) # 16 
        # x = self.layer4_1(self.actvn(x))
        x = self.layer5(self.actvn(x)) # 8
        
        '''x_global = self.global_avg(x)
        x_global = x_global.view(x.size(0), -1)
        x_global = self.fc_0(x_global)
        x_global = self.fc_1(self.actvn(x_global))
        x_global = x_global.view(x.size(0), self.ndf, 4, 4)'''
        x = self.layer6(self.actvn(x)) # 8
        
        '''x = self.layer7(self.actvn(x))
        x = self.layer8(self.actvn(x)) # 4
        # print(x.shape) # 16 * 16 
        out = self.layer8_1(x)'''

        '''mu = self.actvn(self.layer8_1(x))
        mu = self.layer8_2(mu)
        logvar = self.actvn(self.layer9_1(x))
        logvar = self.layer9_2(logvar)'''
        x = self.res_0(x)
        x = self.res_1(x)
        x = self.res_2(x)
        x = self.down(x)
        x = self.res_3(x)
        x = self.res_4(x)
        mu = self.out(self.pad_1(x))

        return mu

class ConvEncoderLoss(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = 64
        self.ndf = ndf
        norm_E = "spectralinstance"
        norm_layer = get_nonspade_norm_layer(None, norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        # self.layer2_1 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        # self.layer3_1 = norm_layer(nn.Conv2d(ndf * 4, ndf * 4, kw, stride=1, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        # self.layer4_1 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))
        # self.layer7 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        # self.layer8 = norm_layer(nn.Conv2d(ndf * 2, ndf * 2, kw, stride=1, padding=pw))
        self.so = s0 = 4
        self.out = norm_layer(nn.Conv2d(ndf * 8, ndf * 4, kw, stride=1, padding=0))
        self.down = nn.AvgPool2d(2,2)
        # self.global_avg = nn.AdaptiveAvgPool2d((6,6))
        
        

        self.actvn = nn.LeakyReLU(0.2, False)
        self.pad_3 = nn.ReflectionPad2d(3)
        self.pad_1 = nn.ReflectionPad2d(1)
        self.conv_7x7 = nn.Conv2d(ndf, ndf, kernel_size=7, padding=0, bias=True)
        # self.opt = opt

    def forward(self, x):

        x1 = self.layer1(x) # 128
        x2 = self.conv_7x7(self.pad_3(self.actvn(x1)))
        x3 = self.layer2(self.actvn(x2)) # 64
        # x = self.layer2_1(self.actvn(x))
        x4 = self.layer3(self.actvn(x3)) # 32
        # x = self.layer3_1(self.actvn(x))
        x5 = self.layer4(self.actvn(x4)) # 16 
        # x = self.layer4_1(self.actvn(x))
        return [x1, x2, x3, x4, x5]
class EncodeMap(BaseNetwork):
    """ Same architecture as the image discriminator """

    def __init__(self, opt):
        super().__init__()

        kw = 3
        pw = int(np.ceil((kw - 1.0) / 2))
        ndf = opt.ngf
        norm_layer = get_nonspade_norm_layer(opt, opt.norm_E)
        self.layer1 = norm_layer(nn.Conv2d(3, ndf, kw, stride=2, padding=pw))
        self.layer2 = norm_layer(nn.Conv2d(ndf * 1, ndf * 2, kw, stride=2, padding=pw))
        self.layer3 = norm_layer(nn.Conv2d(ndf * 2, ndf * 4, kw, stride=2, padding=pw))
        self.layer4 = norm_layer(nn.Conv2d(ndf * 4, ndf * 8, kw, stride=2, padding=pw))
        self.layer5 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=2, padding=pw))
        if opt.crop_size >= 256:
            self.layer6 = norm_layer(nn.Conv2d(ndf * 8, ndf * 8, kw, stride=1, padding=pw))

        self.so = s0 = 4
        self.fc_mu = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.fc_var = nn.Linear(ndf * 8 * s0 * s0, 256)
        self.layer_final = nn.Conv2d(ndf * 8, ndf * 16, kw, stride=1, padding=pw)

        self.actvn = nn.LeakyReLU(0.2, False)
        self.opt = opt

    def forward(self, x):
        if x.size(2) != 256 or x.size(3) != 256:
            x = F.interpolate(x, size=(256, 256), mode='bilinear')

        x = self.layer1(x)
        x = self.layer2(self.actvn(x))
        x = self.layer3(self.actvn(x))
        x = self.layer4(self.actvn(x))
        x = self.layer5(self.actvn(x))
        # if self.opt.crop_size >= 256:
        #     x = self.layer6(self.actvn(x))
        x = self.actvn(x)
        return self.layer_final(x)

        x = x.view(x.size(0), -1)
        mu = self.fc_mu(x)
        logvar = self.fc_var(x)

        return mu, logvar