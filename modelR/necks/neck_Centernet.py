import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.convolutions import Convolutional

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
        """
        x1 means previous output; x2 means current output
        """
        out = torch.cat((x2, x1), dim=1)
        return out

class Neck(nn.Module):
    def __init__(self, fileters_in, fileters_out):
        super(Neck, self).__init__()

        fi_0, fi_1, fi_2, fi_3 = fileters_in

        #32
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1,pad=0, norm="bn", activate="leaky"),
        )
        self.__conv0 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route0 = Route()

        #16
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1+256, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )
        self.__conv1 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route1 = Route()

        #8
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2+128, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            #Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )
        self.__conv2 = Convolutional(filters_in=128, filters_out=64, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample2 = Upsample(scale_factor=2)
        self.__route2 = Route()

        #4
        self.__conv_set_3 = nn.Sequential(
            Convolutional(filters_in=fi_3+64, filters_out=64, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=64, filters_out=128, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=64, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=64, filters_out=128, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
        )

        self.xy_pred = Convolutional(filters_in=128, filters_out=2, kernel_size=1, stride=1, pad=0)

        self.wh_pred = Convolutional(filters_in=128, filters_out=2, kernel_size=1, stride=1, pad=0)

        self.ar_pred = Convolutional(filters_in=128, filters_out=5, kernel_size=1, stride=1, pad=0)

        self.cls_pred = Convolutional(filters_in=128, filters_out=fileters_out-9, kernel_size=1, stride=1, pad=0)

    def forward(self, x0, x1, x2, x3):  # large, medium, small
        #32
        r0 = self.__conv_set_0(x0)
        #16
        r1 = self.__upsample0(self.__conv0(r0))
        r1 = self.__route0(x1, r1)
        r1 = self.__conv_set_1(r1)
        #8
        r2 = self.__conv1(self.__upsample1(r1))
        r2 = self.__route1(x2, r2)
        r2 = self.__conv_set_2(r2)
        #4
        r3 = self.__conv2(self.__upsample1(r2))
        r3 = self.__route2(x3, r3)
        r3 = self.__conv_set_3(r3)

        xy_pred = self.xy_pred(r3)
        wh_pred = self.wh_pred(r3)
        ar_pred = self.ar_pred(r3)
        cls_pred = self.cls_pred(r3)
        out = torch.cat((xy_pred, wh_pred, ar_pred, cls_pred), dim=1)
        return out