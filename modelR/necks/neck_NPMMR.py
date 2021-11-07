import torch
import torch.nn as nn
import torch.nn.functional as F
import config.config as cfg
from ..layers.convolutions import Convolutional, Deformable_Convolutional
from ..layers.msr_blocks import MSR_L, MSR_M, MSR_S
from ..head.head_NPMMR import MTR_Head1

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

class MSR_FPN(nn.Module):
    def __init__(self, fileters_in):

        super(MSR_FPN, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in
        self.__fo = cfg.DATA["NUM"]

        self.__conv21down = Deformable_Convolutional(filters_in=fi_2, filters_out=256, kernel_size=3,
                                                     stride=2, pad=1, norm="bn", activate="leaky")
        self.__route21 = Route()
        self.__conv10down = Deformable_Convolutional(filters_in=fi_1 + 256, filters_out=512, kernel_size=3,
                                                     stride=2, pad=1, norm="bn", activate="leaky")

        self.__route10 = Route()
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0 + 512, filters_out=512, kernel_size=1,
                          stride=1, pad=0, norm="bn", activate="Mish"),
            MSR_L(512),
        )
        self.__conv01up = Convolutional(filters_in=512, filters_out=256, kernel_size=1,
                                        stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route01 = Route()

        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1 + 256 + 256, filters_out=256, kernel_size=1,
                          stride=1, pad=0, norm="bn", activate="Mish"),
            MSR_M(256),
        )
        self.__conv12up = Convolutional(filters_in=256, filters_out=128, kernel_size=1,
                                        stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route12 = Route()

        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2 + 128, filters_out=128, kernel_size=1,
                          stride=1, pad=0, norm="bn", activate="Mish"),
            MSR_S(128),
        )

        self.__conv0_1 = MTR_Head1(filters_in=512, fo_class=self.__fo, temp=False)
        self.__conv1_1 = MTR_Head1(filters_in=256, fo_class=self.__fo, temp=False)
        self.__conv2_1 = MTR_Head1(filters_in=128, fo_class=self.__fo, temp=False)

    def forward(self, x0, x1, x2):
        conv21down = self.__conv21down(x2)
        route21 = self.__route21(x1, conv21down)
        conv10down = self.__conv10down(route21)
        route10 = self.__route10(x0, conv10down)
        conv_set_0 = self.__conv_set_0(route10)

        conv01up = self.__conv01up(conv_set_0)
        upsample0 = self.__upsample0(conv01up)
        route01 = self.__route01(route21, upsample0)
        conv_set_1 = self.__conv_set_1(route01)

        conv12up = self.__conv12up(conv_set_1)
        upsample1 = self.__upsample1(conv12up)
        route12 = self.__route12(x2, upsample1)
        conv_set_2 = self.__conv_set_2(route12)

        out0 = self.__conv0_1(conv_set_0)
        out1 = self.__conv1_1(conv_set_1)
        out2 = self.__conv2_1(conv_set_2)

        return out2, out1, out0  # small, medium, large