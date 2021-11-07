import torch
torch.cuda.current_device()
import torch.nn as nn
import torch.nn.functional as F
from ..layers.convolutions import Convolutional
from dropblock import DropBlock2D, LinearScheduler
from ..layers.multiscale_fusion_blocks import SPP
from ..layers.msr_blocks import MSR_L, MSR_M, MSR_S

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
        fi_0, fi_1, fi_2 = fileters_in
        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            SPP(depth=512),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )

        self.__conv0_xywharconf = nn.Sequential(
            #Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #Convolutional(filters_in=1024, filters_out=512, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=1024, out_channels=5, kernel_size=1, stride=1, padding=0),
        )

        self.__conv0_cls = nn.Sequential(
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=1024, out_channels=fileters_out - 5, kernel_size=1, stride=1, padding=0),
        )

        self.__conv0 = Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route0 = Route()

        # medium
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1+256, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )

        self.__conv1_xywharconf = nn.Sequential(
            #Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #Convolutional(filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=512, out_channels=5, kernel_size=1, stride=1, padding=0),
        )

        self.__conv1_cls = nn.Sequential(
            Convolutional(filters_in=256, filters_out=512, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=512, out_channels=fileters_out - 5, kernel_size=1, stride=1, padding=0),
        )

        self.__conv1 = Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route1 = Route()

        # small
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2+128, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
        )

        self.__conv2_xywharconf = nn.Sequential(
            #Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            #Convolutional(filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=256, out_channels=5, kernel_size=1, stride=1, padding=0),
        )

        self.__conv2_cls = nn.Sequential(
            Convolutional(filters_in=128, filters_out=256, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=256, out_channels=fileters_out - 5, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x0, x1, x2):  # large, medium, small
        # large
        r0 = self.__conv_set_0(x0)
        #t0 = self.__conv0_0(r0)
        out0_xywharconf = self.__conv0_xywharconf(r0)
        out0_cls = self.__conv0_cls(r0)
        out0 = torch.cat((out0_xywharconf, out0_cls), dim=1)

        # medium
        r1 = self.__conv0(r0)
        r1 = self.__upsample0(r1)
        x1 = self.__route0(x1, r1)
        r1 = self.__conv_set_1(x1)
        out1_xywharconf = self.__conv1_xywharconf(r1)
        out1_cls = self.__conv1_cls(r1)
        out1 = torch.cat((out1_xywharconf, out1_cls), dim=1)

        # small
        r2 = self.__conv1(r1)
        r2 = self.__upsample1(r2)
        x2 = self.__route1(x2, r2)
        r2 = self.__conv_set_2(x2)
        out2_xywharconf = self.__conv2_xywharconf(r2)
        out2_cls = self.__conv2_cls(r2)
        out2 = torch.cat((out2_xywharconf, out2_cls), dim=1)

        return out2, out1, out0  # small, medium, large

class MSR_FPN4(nn.Module):
    def __init__(self, fileters_in, fileters_out):
        super(MSR_FPN4, self).__init__()
        fi_0, fi_1, fi_2, fi_3 = fileters_in

        self.__conv21down = Convolutional(
            filters_in=fi_2, filters_out=256, kernel_size=3, stride=2, pad=1, norm="bn", activate="leaky")
        self.__route21 = Route()
        self.__conv10down = Convolutional(
            filters_in=fi_1 + 256, filters_out=512, kernel_size=3, stride=2, pad=1, norm="bn", activate="leaky")
#####################################################################################
        self.__route10 = Route()
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0 + 512, filters_out=512,
            kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            MSR_L(512),
        )
        self.__conv01up = Convolutional(
            filters_in=512, filters_out=256, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample0 = Upsample(scale_factor=2)
        self.__route01 = Route()
#######################################################################################
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fi_1 + 256 + 256, filters_out=256,
            kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            MSR_M(256),
        )
        self.__conv12up = Convolutional(
            filters_in=256, filters_out=128, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample1 = Upsample(scale_factor=2)
        self.__route12 = Route()
########################################################################################
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fi_2 + 128, filters_out=128,
            kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            MSR_S(128),
        )
        self.__conv23up = Convolutional(
            filters_in=128, filters_out=64, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__upsample2 = Upsample(scale_factor=2)
        self.__route23 = Route()
########################################################################################
        self.__conv_set_3 = nn.Sequential(
            Convolutional(filters_in=fi_3 + 64, filters_out=64,
            kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=64, filters_out=128, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=64, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=64, filters_out=128, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=64, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
        )
################################################################################################
        ######################################################

        self.__loc0 = nn.Sequential(
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=1024, out_channels=5,
            kernel_size=1, stride=1, padding=0),
        )

        self.__cls0 = nn.Sequential(
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=1024, filters_out=512, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=512, filters_out=1024, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=1024, out_channels=fileters_out -
            5, kernel_size=1, stride=1, padding=0),
        )
###################################################################################################
        self.__loc1 = nn.Sequential(
            Convolutional(filters_in=256, filters_out=512, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=512, out_channels=5,
            kernel_size=1, stride=1, padding=0),
        )
        self.__cls1 = nn.Sequential(
            Convolutional(filters_in=256, filters_out=512, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=512, filters_out=256, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=512, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=512, out_channels=fileters_out -
            5, kernel_size=1, stride=1, padding=0),
        )
##################################################################################################
        self.__loc2 = nn.Sequential(
            Convolutional(filters_in=128, filters_out=256, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=256, out_channels=5,
            kernel_size=1, stride=1, padding=0),
        )
        self.__cls2 = nn.Sequential(
            Convolutional(filters_in=128, filters_out=256, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=256, filters_out=128, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=256, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=256, out_channels=fileters_out -
            5, kernel_size=1, stride=1, padding=0),
        )
######################################################################################################
        self.__locxs = nn.Sequential(
            Convolutional(filters_in=64, filters_out=128, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=64, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=64, filters_out=128, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=128, out_channels=5,
            kernel_size=1, stride=1, padding=0),
        )

        self.__clsxs = nn.Sequential(
            Convolutional(filters_in=64, filters_out=128, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            Convolutional(filters_in=128, filters_out=64, kernel_size=1,
            stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=64, filters_out=128, kernel_size=3,
            stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=128, out_channels=fileters_out -
            5, kernel_size=1, stride=1, padding=0),
        )

    def forward(self, x0, x1, x2, x3):
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

        conv23up = self.__conv23up(conv_set_2)
        upsample2 = self.__upsample2(conv23up)
        route23 = self.__route23(x3, upsample2)
        conv_set_xs = self.__conv_set_3(route23)

###############################################################
        loc0 = self.__loc0(conv_set_0)
        cls0 = self.__cls0(conv_set_0)
        out_0 = torch.cat((loc0, cls0), dim=1)

        loc1 = self.__loc1(conv_set_1)
        cls1 = self.__cls1(conv_set_1)
        out_1 = torch.cat((loc1, cls1), dim=1)

        loc2 = self.__loc2(conv_set_2)
        cls2 = self.__cls2(conv_set_2)
        out_2 = torch.cat((loc2, cls2), dim=1)

        locxs = self.__locxs(conv_set_xs)
        clsxs = self.__clsxs(conv_set_xs)
        out_xs = torch.cat((locxs, clsxs), dim=1)

        return out_xs, out_2, out_1, out_0  # small, medium, large
