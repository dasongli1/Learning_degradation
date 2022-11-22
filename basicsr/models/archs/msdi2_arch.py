import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
from basicsr.models.archs.architecture1 import SPADEResnetBlock1 as SPADEResnetBlock
from basicsr.models.archs.encoder2 import ConvEncoder
from collections import OrderedDict
from copy import deepcopy

def conv3x3(in_chn, out_chn, bias=True):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=3, stride=1, padding=1, bias=bias)
    return layer

def conv_down(in_chn, out_chn, bias=False):
    layer = nn.Conv2d(in_chn, out_chn, kernel_size=4, stride=2, padding=1, bias=bias)
    return layer

def conv(in_channels, out_channels, kernel_size, bias=False, stride = 1):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size//2), bias=bias, stride = stride)

## Supervised Attention Module
class SAM(nn.Module):
    def __init__(self, n_feat, kernel_size=3, bias=True):
        super(SAM, self).__init__()
        self.conv1 = conv(n_feat, n_feat, kernel_size, bias=bias)
        self.conv2 = conv(n_feat, 3, kernel_size, bias=bias)
        self.conv3 = conv(3, n_feat, kernel_size, bias=bias)

    def forward(self, x, x_img):
        x1 = self.conv1(x)
        img = self.conv2(x) + x_img
        x2 = torch.sigmoid(self.conv3(img))
        x1 = x1*x2
        x1 = x1+x
        return x1, img
class Up_ConvBlock(nn.Module):
    def __init__(self, dim_in, dim_out, activation=nn.LeakyReLU(0.2,False), kernel_size=3):
        super().__init__()

        pw = (kernel_size - 1) // 2
        '''self.conv1 = nn.Sequential(
            nn.ReflectionPad2d(pw),
            nn.Conv2d(dim, dim, kernel_size=kernel_size),
            activation)'''
        # norm_layer = 
        self.conv_block = nn.Sequential(
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_in, dim_out, kernel_size=kernel_size)),
            activation,
            nn.Upsample(scale_factor=2),
            nn.ReflectionPad2d(pw),
            spectral_norm(nn.Conv2d(dim_out, dim_out, kernel_size=kernel_size)),
            activation
        )
        

    def forward(self, x):
        # conv1 = self.conv1(x)
        y = self.conv_block(x)
        return y
    
class prior_upsampling(nn.Module):
    def __init__(self, wf=64):
        super(prior_upsampling, self).__init__()
        # self.conv_latent_init = Up_ConvBlock(4 * wf, 32 * wf)
        self.conv_latent_up2 = Up_ConvBlock(32 * wf, 16 * wf) 
        self.conv_latent_up3 = Up_ConvBlock(16 * wf, 8 * wf)
        self.conv_latent_up4 = Up_ConvBlock(8 * wf, 4 * wf)
        self.conv_latent_up5 = Up_ConvBlock(4 * wf, 2 * wf)
        self.conv_latent_up6 = Up_ConvBlock(2 * wf, 1 * wf)
    def forward(self, z):
        # latent_1 = self.conv_latent_init(z) # 8, 8
        latent_2 = self.conv_latent_up2(z) # 16
        latent_3 = self.conv_latent_up3(latent_2) # 32
        latent_4 = self.conv_latent_up4(latent_3) # 64
        latent_5 = self.conv_latent_up5(latent_4) # 128
        latent_6 = self.conv_latent_up6(latent_5) # 256
        latent_list = [latent_6,latent_5,latent_4,latent_3]
        return latent_list # latent_6,latent_5,latent_4,latent_3,latent_2,latent_1

# for k,v in isp.named_parameters():
#     v.requires_grad=False

class msdi2_net(nn.Module):

    def __init__(self, in_chn=3, wf=64, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(msdi2_net, self).__init__()
        self.depth = depth
        self.down_path_1 = nn.ModuleList()
        self.down_path_2 = nn.ModuleList()
        self.conv_01 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.conv_02 = nn.Conv2d(in_chn, wf, 3, 1, 1)
        self.prior_upsampling = prior_upsampling()
        # self.prior_upsampling.load_state_dict(torch.load("/workspace/lidasong/HINet/experiments/GoPro-HINet7hg17L/models/net_g_200000.pth")['params'])
        # state_dict = torch.load("/workspace/lidasong/1025/HINet/experiments/GoPro-HINet5hg6L8/models/net_g_latest.pth")['params']
        # print(type(state_dict))
        net_prior_dict = torch.load("/workspace/lidasong/Learning_degradation/checkpoints/net2.pth")
        prior_upsampling_dict = torch.load("/workspace/lidasong/Learning_degradation/checkpoints/prior_upsampling2.pth")
        
        
        #for ele in state_dict:
        #    print(ele)
        #prior_upsampling_dict = OrderedDict()
        #net_prior_dict = OrderedDict()
        #for k, v in deepcopy(state_dict).items():
        #    if k.startswith('prior_upsampling.'):
        #        prior_upsampling_dict[k[len('prior_upsampling.'):]] = v
        #    if k.startswith('net_prior.'):
        #        net_prior_dict[k[len('net_prior.'):]] = v
        self.prior_upsampling.load_state_dict(prior_upsampling_dict, strict=True)
        # exit(0)

        self.net_prior = ConvEncoder()
        # self.net_prior.load_state_dict(torch.load("/workspace/lidasong/HINet/experiments/GoPro-HINet7hg17L/models/net_g_200000.pth"), strict=False)
        self.net_prior.load_state_dict(net_prior_dict, strict=True)
        # exit(0)
        
        #torch.save(self.net_prior.state_dict(), "net2.pth")
        #torch.save(self.prior_upsampling.state_dict(), "prior_upsampling2.pth")

        for k,v in self.prior_upsampling.named_parameters():
            v.requires_grad=False
        for k,v in self.net_prior.named_parameters():
            v.requires_grad=False
        # del state_dict
        del prior_upsampling_dict
        del net_prior_dict 
        torch.cuda.empty_cache()

        self.ad1_list = nn.ModuleList()
        # self.ad2_list = nn.ModuleList()

        prev_channels = self.get_input_chn(wf)
        norm_G = "spectralspadesyncbatch3x3"
        for i in range(depth): #0,1,2,3,4
            use_HIN = True if hin_position_left <= i and i <= hin_position_right else False
            downsample = True if (i+1) < depth else False
            self.down_path_1.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_HIN=use_HIN))
            self.down_path_2.append(UNetConvBlock(prev_channels, (2**i) * wf, downsample, relu_slope, use_csff=downsample, use_HIN=use_HIN))
            self.ad1_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, norm_G, label_nc=(2**i) * wf))
            # self.ad2_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, norm_G, label_nc=(2**i) * wf))
            '''if i == 0:
                self.ad1_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, opt, label_nc=(2**i) * wf))
                # self.ad2_list.append(SPADEResnetBlock((2**i) * wf, (2**i) * wf, opt, label_nc=(2**i) * wf))
            else:
                self.ad1_list.append(SPADEResnetBlock((2**(i-1)) * wf, (2**(i-1)) * wf, opt, label_nc=(2**(i-1)) * wf))
                # self.ad2_list.append(SPADEResnetBlock((2**(i-1)) * wf, (2**(i-1)) * wf, opt, label_nc=(2**(i-1)) * wf))'''
            prev_channels = (2**i) * wf
        self.ad1_list = self.ad1_list[0:-1]
        # self.ad2_list = self.ad2_list[0:-1]
        self.up_path_1 = nn.ModuleList()
        self.up_path_2 = nn.ModuleList()
        self.skip_conv_1 = nn.ModuleList()
        self.skip_conv_2 = nn.ModuleList()
        self.skip_conv_3 = nn.ModuleList()
        self.skip_conv_4 = nn.ModuleList()
        self.skip_conv_5 = nn.ModuleList()
        self.skip_conv_6 = nn.ModuleList()
        for i in reversed(range(depth - 1)):
            self.up_path_1.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            self.up_path_2.append(UNetUpBlock(prev_channels, (2**i)*wf, relu_slope))
            # self.skip_conv_1.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            # self.skip_conv_2.append(nn.Conv2d((2**i)*wf, (2**i)*wf, 3, 1, 1))
            self.skip_conv_1.append(UNetConvBlock((2**i) * wf, (2**i) * wf, False, relu_slope, use_HIN=True))
            self.skip_conv_2.append(UNetConvBlock((2**i) * wf, (2**i) * wf, False, relu_slope, use_HIN=True))
            self.skip_conv_3.append(UNetConvBlock((2**i) * wf, (2**i) * wf, False, relu_slope, use_HIN=True))
            self.skip_conv_4.append(UNetConvBlock((2**i) * wf, (2**i) * wf, False, relu_slope, use_HIN=True))
            self.skip_conv_5.append(UNetConvBlock((2**i) * wf, (2**i) * wf, False, relu_slope, use_HIN=True))
            self.skip_conv_6.append(UNetConvBlock((2**i) * wf, (2**i) * wf, False, relu_slope, use_HIN=True))
            
            prev_channels = (2**i)*wf
        self.sam12 = SAM(prev_channels)
        self.cat12 = nn.Conv2d(prev_channels*2, prev_channels, 1, 1, 0)

        self.last = conv3x3(prev_channels, in_chn, bias=True)
        # 
        # self.conv_latent_init = nn.Conv2d(4 * wf, 32 * wf, 3, padding=1)
        '''self.conv_latent_init = Up_ConvBlock(4 * wf, 32 * wf)
        self.conv_latent_up2 = Up_ConvBlock(32 * wf, 16 * wf) 
        self.conv_latent_up3 = Up_ConvBlock(16 * wf, 8 * wf)
        self.conv_latent_up4 = Up_ConvBlock(8 * wf, 4 * wf)
        self.conv_latent_up5 = Up_ConvBlock(4 * wf, 2 * wf)
        self.conv_latent_up6 = Up_ConvBlock(2 * wf, 1 * wf)'''
        
        # self.ad6 = SPADEResnetBlock(16 * wf, 16 * wf, opt)
        

    def forward(self, x):
        prior_z = self.net_prior(x)
        latent_list = self.prior_upsampling(prior_z)
        # latent_list = [latent_6,latent_5,latent_4,latent_3,latent_2,latent_1]
        image = x
        #stage 1
        x1 = self.conv_01(image)
        encs = []
        decs = []
        # print("x1", x1.shape)
        for i, down in enumerate(self.down_path_1):
            if (i+1) < self.depth:
                # print("i--spade", i, x1.shape, latent_list[i].shape)
                # x1 = self.ad1_list[i](x1, latent_list[i])
                # print("i--spade output", i, x1.shape)
                x1, x1_up = down(x1) # 64, 128, 128 -- 64, 256, 256
                # print("i", i, x1.shape, x1_up.shape)
                
                encs.append(x1_up)
            else:
                # print("i", i, x1.shape, latent_list[i].shape)
                # x1 = self.ad1_list[i](x1, latent_list[i])
                # print("i spade", i, x1.shape)
                x1 = down(x1) # 2048, 8, 8
                # print("i - nodown", i, x1.shape)
                # x1 = self.ad1_list[-1](x1, latent_list[-1])
                

        for i, up in enumerate(self.up_path_1):
            temps = self.skip_conv_1[i](encs[-i-1])
            # (8,8) ---- (1024,16,16) --- (16,16)
            # print("i temps2 input", i, encs[-i-1].shape, latent_list[-2-i].shape)

            temps2 = self.ad1_list[-1-i](temps, latent_list[-1-i])
            temps2 = self.skip_conv_3[i](temps2)
            # print("i, temps shape", i, x1.shape, encs[-i-1].shape, temps.shape, temps2.shape)
            x1 = up(x1, temps2)
            decs.append(x1)

        sam_feature, out_1 = self.sam12(x1, image)
        # return out_1
        
        #stage 2
        x2 = self.conv_02(image)
        x2 = self.cat12(torch.cat([x2, sam_feature], dim=1))
        blocks = []
        for i, down in enumerate(self.down_path_2):
            if (i+1) < self.depth:
                # x2 = self.ad2_list[i](x2, latent_list[i])
                # print("x2--spade", i, x2.shape, latent_list[i].shape)
                # x2 = self.ad2_list[i](x2, latent_list[i])
                # print("x2 spade output", i, x2.shape)
                x2, x2_up = down(x2, encs[i], decs[-i-1])
                # print("x2 out", i, x2.shape, x2_up.shape)
                blocks.append(x2_up)
            else:
                # print
                # x2 = self.ad2_list[i](x2, latent_list[i])
                x2 = down(x2)
                # x2 = self.ad2_list[-1](x2, latent_list[-1])
                # print("x2 out", i, x2.shape)

        for i, up in enumerate(self.up_path_2):
            temps = self.skip_conv_2[i](blocks[-i-1])
            # print("i temps2 input", i, blocks[-i-1].shape, latent_list[-2-i].shape)
            # temps2 = self.ad2_list[-1-i](temps, latent_list[-1-i])
            temps2 = self.skip_conv_4[i](temps)
            temps2 = self.skip_conv_5[i](temps2)
            temps2 = self.skip_conv_6[i](temps2)

            # print("i, temps shape", i, x1.shape, temps2.shape)
            x2 = up(x2, temps2)

        out_2 = self.last(x2)
        out_2 = out_2 + image
        return [out_1, out_2]

    def get_input_chn(self, in_chn):
        return in_chn

    def _initialize(self):
        gain = nn.init.calculate_gain('leaky_relu', 0.20)
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.orthogonal_(m.weight, gain=gain)
                if not m.bias is None:
                    nn.init.constant_(m.bias, 0)


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, downsample, relu_slope, use_csff=False, use_HIN=False):
        super(UNetConvBlock, self).__init__()
        self.downsample = downsample
        self.identity = nn.Conv2d(in_size, out_size, 1, 1, bias=False)
        self.use_csff = use_csff
        # ratio = 8

        self.conv_1 = nn.Conv2d(in_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_1 = nn.LeakyReLU(relu_slope, inplace=False)
        self.conv_2 = nn.Conv2d(out_size, out_size, kernel_size=3, padding=1, bias=True)
        self.relu_2 = nn.LeakyReLU(relu_slope, inplace=False)
        
        self.conv_3 = nn.Conv2d(out_size, out_size, kernel_size=1, padding=0, bias=False)
        self.conv_4 = nn.Conv2d(out_size, out_size, kernel_size=1, padding=0, bias=False)


        if downsample and use_csff:
            self.csff_enc = nn.Conv2d(out_size, out_size, 1, 1, bias=False)
            self.csff_dec = nn.Conv2d(out_size, out_size, 1, 1, bias=False)

        if use_HIN:
            self.norm = nn.InstanceNorm2d(out_size//2, affine=True)
        self.use_HIN = use_HIN

        if downsample:
            self.downsample = conv_down(out_size, out_size, bias=False)

    def forward(self, x, enc=None, dec=None):
        out = self.conv_1(x)

        if self.use_HIN:
            out_1, out_2 = torch.chunk(out, 2, dim=1)
            out = torch.cat([self.norm(out_1), out_2], dim=1)
        out = self.relu_1(out)
        out = self.relu_2(self.conv_3(out))
        out = self.relu_2(self.conv_2(out))
        # out = self.conv_4(out)
        # out = self.relu_2(self.conv_3(out))
        # out = self.conv_4(out)

        out += self.identity(x)
        if enc is not None and dec is not None:
            assert self.use_csff
            out = out + self.csff_enc(enc) + self.csff_dec(dec)
        if self.downsample:
            out_down = self.downsample(out)
            return out_down, out
        else:
            return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, relu_slope):
        super(UNetUpBlock, self).__init__()
        self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2, bias=True)
        self.conv_block = UNetConvBlock(in_size, out_size, False, relu_slope)

    def forward(self, x, bridge):
        up = self.up(x)
        out = torch.cat([up, bridge], 1)
        out = self.conv_block(out)
        return out

class Subspace(nn.Module):

    def __init__(self, in_size, out_size):
        super(Subspace, self).__init__()
        self.blocks = nn.ModuleList()
        self.blocks.append(UNetConvBlock(in_size, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for i in range(len(self.blocks)):
            x = self.blocks[i](x)
        return x + sc


class skip_blocks(nn.Module):

    def __init__(self, in_size, out_size, repeat_num=1):
        super(skip_blocks, self).__init__()
        self.blocks = nn.ModuleList()
        self.re_num = repeat_num
        mid_c = 128
        self.blocks.append(UNetConvBlock(in_size, mid_c, False, 0.2))
        for i in range(self.re_num - 2):
            self.blocks.append(UNetConvBlock(mid_c, mid_c, False, 0.2))
        self.blocks.append(UNetConvBlock(mid_c, out_size, False, 0.2))
        self.shortcut = nn.Conv2d(in_size, out_size, kernel_size=1, bias=True)

    def forward(self, x):
        sc = self.shortcut(x)
        for m in self.blocks:
            x = m(x)
        return x + sc

# msdi = msdi2_net()
