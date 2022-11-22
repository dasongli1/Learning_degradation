import torch
import torch.nn as nn
import torch.nn.utils.spectral_norm as spectral_norm
# from basicsr.models.archs.architecture16 import SPADEResnetBlock1 as SPADEResnetBlock
from basicsr.models.archs.encoder2 import ConvEncoder
from basicsr.models.archs.generator import HINet


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

class MSDI2E(nn.Module):

    def __init__(self, in_chn=3, wf=64, depth=5, relu_slope=0.2, hin_position_left=0, hin_position_right=4):
        super(MSDI2E, self).__init__()
        self.depth = depth
        self.prior_upsampling = prior_upsampling()
        # self.prior_upsampling.load_state_dict(torch.load("/workspace/lidasong/SPADE/checkpoints/generator37/latest_net_G.pth"), strict=False)
        # ,"/workspace/lidasong/HINet/prior_latest_encoder37.pth"), strict=False)
        self.net_prior = ConvEncoder()
        # self.net_prior.load_state_dict(torch.load("/workspace/lidasong/SPADE/checkpoints/generator37/latest_net_E.pth"), strict=False)
        # exit(0)
                    
        self.inverse_generator = HINet()
        self.generator = HINet()
        
        

    def forward(self, x, y):
        prior_z = self.net_prior(x)
        latent_list_inverse = self.prior_upsampling(prior_z)
        # latent_list = [latent_6,latent_5,latent_4,latent_3,latent_2,latent_1]
        out_inverse = self.inverse_generator(y, latent_list_inverse)
        out = self.generator(x, latent_list_inverse)
        # print(out_inverse[0].shape)
        # exit(0)
        return out, out_inverse

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
