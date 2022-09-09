import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.multiscale_fusion_blocks import SPP
from ..layers.convolutions import Convolutional
from ..layers.msr_blocks import MSR_L, MSR_M, MSR_S
from ..layers.np_attention_blocks import NPAttention1
import config.config as cfg
from dropblock import DropBlock2D, LinearScheduler
from ..layers.ses_conv import SESConvolutional
from ..layers.enn import E2Convolutional, C8SteerableCNN

class Upsample(nn.Module):
    def __init__(self, scale_factor=1, mode='nearest'):
        super(Upsample, self).__init__()
        self.scale_factor = scale_factor
        self.mode = mode

    def forward(self, x):
        return F.interpolate(x, scale_factor=self.scale_factor, mode=self.mode)


class Route(nn.Module):
    def __init__(self):
        super(Route, self).__init__()

    def forward(self, x1, x2):
        out = torch.cat((x2, x1), dim=1)
        return out

class Neck(nn.Module):
    def __init__(self, fileters_in, fileters_out):
        super(Neck, self).__init__()
        fi_0, fi_1, fi_2, fi_3 = fileters_in
        self.scales_per_layer = cfg.MODEL["SCALES_PER_LAYER"]
        #self.__np0 = NPAttention1(fi_0, fi_1, use_scale=False, groups=8)
        #self.__np1 = NPAttention1(fi_1, fi_2, use_scale=False, groups=8)
        #self.__np2 = NPAttention1(fi_2, fi_3, use_scale=False, groups=8)
        #self.__np3 = NPAttention1(128, 64, use_scale=False, groups=8)
        #self.__s0 = SESConvolutional(fi_0, fi_0)
        #self.__s1 = SESConvolutional(fi_1, fi_1)
        #self.__s2 = SESConvolutional(fi_2, fi_2)

        self.fileters_out = fileters_out
        #self.__conv21down = Convolutional(filters_in=fi_2, filters_out=fi_2, kernel_size=3, stride=2, pad=1, norm="bn", activate="leaky")
        #self.__route21 = Route()
        #self.__conv10down = Convolutional(filters_in=fi_1 + fi_2, filters_out=fi_1, kernel_size=3, stride=2, pad=1, norm="bn", activate="leaky")
#####################################################################################
        #self.__route10 = Route()
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #NPAttention1(fi_1, fi_2, use_scale=False, groups=8),
            SPP(fi_1),
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #MSR_L(fi_1),
        )
        self.__conv01up = Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route01 = Route()
#######################################################################################
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1 + fi_2, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #NPAttention1(fi_2, fi_3, use_scale=False, groups=8),
            #MSR_M(fi_2),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )
        self.__conv12up = Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route12 = Route()
########################################################################################
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2 + fi_3, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #NPAttention1(fi_3, fi_3//2, use_scale=False, groups=8),
            #MSR_S(fi_3),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )
        '''
        self.__down21 = Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=2, pad=1, norm="bn", activate="leaky")
        self.__route21 = Route()
        self.__conv_set_1a = nn.Sequential(
            Convolutional(filters_in=fi_2 + fi_2, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #NPAttention1(256, 128, use_scale=False, groups=8),
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )

        self.__down10 = Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=2, pad=1, norm="bn", activate="leaky")
        self.__route10 = Route()
        self.__conv_set_0a = nn.Sequential(
            Convolutional(filters_in=fi_1 + fi_1, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #NPAttention1(512, 256, use_scale=False, groups=8),
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )
        ######################################################
        

        self.__r80 = C8SteerableCNN(fi_1, fi_1)
        self.__r81 = C8SteerableCNN(fi_2, fi_2)
        self.__r82 = C8SteerableCNN(fi_3, fi_3)
        self.__s0 = SESConvolutional(fi_1, fi_1)
        self.__s1 = SESConvolutional(fi_2, fi_2)
        self.__s2 = SESConvolutional(fi_3, fi_3)'''

        self.__loc0 = nn.Sequential(
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #E2Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1,stride=1, pad=0, norm="bn", activate="leaky"),
            #Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #nn.Conv2d(in_channels=fi_0, out_channels=10, kernel_size=1, stride=1, padding=0),
        )

        self.__cls0 = nn.Sequential(
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #E2Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #nn.Conv2d(in_channels=fi_0, out_channels=(fileters_out - 10), kernel_size=1, stride=1, padding=0),
        )
###################################################################################################
        self.__loc1 = nn.Sequential(
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #E2Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn",activate="leaky"),
            #Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #nn.Conv2d(in_channels=fi_1, out_channels=10, kernel_size=1, stride=1, padding=0),
        )
        self.__cls1 = nn.Sequential(
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #E2Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn",activate="leaky"),
            #Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #nn.Conv2d(in_channels=fi_1, out_channels=(fileters_out-10), kernel_size=1, stride=1, padding=0),
        )
##################################################################################################
        self.__loc2 = nn.Sequential(
            Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #E2Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn",activate="leaky"),
            #Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #nn.Conv2d(in_channels=fi_2, out_channels=10, kernel_size=1, stride=1, padding=0),
        )
        self.__cls2 = nn.Sequential(
            Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #E2Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn",activate="leaky"),
            #Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #nn.Conv2d(in_channels=fi_2, out_channels=(fileters_out-10), kernel_size=1, stride=1, padding=0),
        )


    def forward(self, x0, x1, x2):
        #x0 = self.__s0(x0)
        #x1 = self.__s1(x1)
        #x2 = self.__s2(x2)
        #x3 = self.__np3(x3)
        #conv21down = self.__conv21down(x2)
        #route21 = self.__route21(x1, conv21down)
        #conv10down = self.__conv10down(route21)
        #route10 = self.__route10(x0, conv10down)
        conv_set_0 = self.__conv_set_0(x0)

        conv01up = self.__conv01up(conv_set_0)
        upsample0 = self.__upsample0(conv01up)
        route01 = self.__route01(x1, upsample0)
        conv_set_1 = self.__conv_set_1(route01)

        conv12up = self.__conv12up(conv_set_1)
        upsample1 = self.__upsample1(conv12up)
        route12 = self.__route12(x2, upsample1)
        conv_set_2 = self.__conv_set_2(route12)

        #down21 = self.__down21(conv_set_2)
        #down21 = self.__route21(conv_set_1, down21)
        #conv_set_1a = self.__conv_set_1a(down21)

        #down10 = self.__down10(conv_set_1a)
        #down10 = self.__route10(conv_set_0, down10)
        #conv_set_0a = self.__conv_set_0a(down10)

###############################################################
        #conv_set_0 = self.__s0(self.__r80(conv_set_0))
        #conv_set_1 = self.__s1(self.__r81(conv_set_1))
        #conv_set_2 = self.__s2(self.__r82(conv_set_2))

        loc0 = self.__loc0(conv_set_0)
        cls0 = self.__cls0(conv_set_0)
        #out_0 = torch.cat((loc0, cls0), dim=1)

        loc1 = self.__loc1(conv_set_1)
        cls1 = self.__cls1(conv_set_1)
        #out_1 = torch.cat((loc1, cls1), dim=1)

        loc2 = self.__loc2(conv_set_2)
        cls2 = self.__cls2(conv_set_2)
        #out_2 = torch.cat((loc2, cls2), dim=1)

        return loc2, cls2, loc1, cls1, loc0, cls0