import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.multiscale_fusion_blocks import SPP
from ..layers.convolutions import Convolutional
from ..layers.msr_blocks import MSR_L, MSR_M, MSR_S
from ..layers.np_attention_blocks import NPAttention1
import config.config as cfg
from dropblock import DropBlock2D, LinearScheduler
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
        self.fileters_out = fileters_out
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            SPP(fi_1),
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_0, filters_out=fi_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )
        self.__conv01up = Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route01 = Route()
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1 + fi_2, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_1, filters_out=fi_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )
        self.__conv12up = Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route12 = Route()
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2 + fi_3, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=fi_2, filters_out=fi_3, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )
        self.__loc0 = nn.Sequential(
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=fi_0, out_channels=10, kernel_size=1, stride=1, padding=0),
        )

        self.__cls0 = nn.Sequential(
            Convolutional(filters_in=fi_1, filters_out=fi_0, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=fi_0, out_channels=(fileters_out - 10), kernel_size=1, stride=1, padding=0),
        )
        self.__loc1 = nn.Sequential(
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=fi_1, out_channels=10, kernel_size=1, stride=1, padding=0),
        )
        self.__cls1 = nn.Sequential(
            Convolutional(filters_in=fi_2, filters_out=fi_1, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=fi_1, out_channels=(fileters_out-10), kernel_size=1, stride=1, padding=0),
        )
        self.__loc2 = nn.Sequential(
            Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=fi_2, out_channels=10, kernel_size=1, stride=1, padding=0),
        )
        self.__cls2 = nn.Sequential(
            Convolutional(filters_in=fi_3, filters_out=fi_2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=fi_2, out_channels=(fileters_out-10), kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x0, x1, x2):
        conv_set_0 = self.__conv_set_0(x0)

        conv01up = self.__conv01up(conv_set_0)
        upsample0 = self.__upsample0(conv01up)
        route01 = self.__route01(x1, upsample0)
        conv_set_1 = self.__conv_set_1(route01)

        conv12up = self.__conv12up(conv_set_1)
        upsample1 = self.__upsample1(conv12up)
        route12 = self.__route12(x2, upsample1)
        conv_set_2 = self.__conv_set_2(route12)

        loc0 = self.__loc0(conv_set_0)
        cls0 = self.__cls0(conv_set_0)
        out_0 = torch.cat((loc0, cls0), dim=1)

        loc1 = self.__loc1(conv_set_1)
        cls1 = self.__cls1(conv_set_1)
        out_1 = torch.cat((loc1, cls1), dim=1)

        loc2 = self.__loc2(conv_set_2)
        cls2 = self.__cls2(conv_set_2)
        out_2 = torch.cat((loc2, cls2), dim=1)

        return out_2, out_1, out_0
