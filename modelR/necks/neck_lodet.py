import torch
import torch.nn as nn
import torch.nn.functional as F
from dropblock import DropBlock2D, LinearScheduler
from modelR.layers.convolutions import Convolutional, Deformable_Convolutional, Cond_Convolutional
import config.config as cfg

class CSA(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3 ,c_tag=0.5, groups=3, dila=1):
        super(CSA, self).__init__()
        self.left_part = round(c_tag * filters_in)
        self.right_part = filters_out - self.left_part
        self.__dw = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=kernel_size, stride=1, pad=(kernel_size-1)//2, groups=self.right_part, dila=dila, norm="bn")
        self.__pw1 = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.groups = groups

    def channel_shuffle(self, features):
        batchsize, num_channels, height, width = features.data.size()
        assert (num_channels % self.groups == 0)
        channels_per_group = num_channels // self.groups
        features = features.view(batchsize, self.groups, channels_per_group, height, width)# reshape
        features = torch.transpose(features, 1, 2).contiguous()
        features = features.view(batchsize, -1, height, width)# flatten
        return features

    def forward(self, x):
        left = x[:, :self.left_part, :, :].contiguous()
        right = x[:, self.left_part:, :, :].contiguous()
        right = self.__dw(right)
        right = self.__pw1(right)
        cat = torch.cat((left, right), 1)
        out = self.channel_shuffle(cat)
        return out

class DRF(nn.Module):
    def __init__(self, filters_in, filters_out, c_tag=0.5, groups=3, dila_r=4, dila_l=6):
        super(DRF, self).__init__()
        self.left_part = round(c_tag * filters_in)
        self.right_part = filters_out - self.left_part
        self.__dw_right = Cond_Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=3,
                                       stride=1, pad=dila_r, groups=self.right_part, dila=dila_r,  bias=True,  norm="bn")
        self.__pw_right = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1,
                                        stride=1, pad=0, norm="bn", activate="leaky")

        self.__dw_left = Cond_Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=3,
                                       stride=1, pad=dila_l, groups=self.right_part, dila=dila_l,  bias=True,  norm="bn")
        self.__pw1_left = Convolutional(filters_in=self.right_part, filters_out=self.right_part, kernel_size=1,
                                        stride=1, pad=0, norm="bn", activate="leaky")

    def forward(self, x):
        left = x[:, :self.left_part, :, :].contiguous()
        right = x[:, self.left_part:, :, :].contiguous()
        left = self.__dw_left(left)
        left = self.__pw1_left(left)
        right = self.__dw_right(right)
        right = self.__pw_right(right)
        return left+right

class CSA_part(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size=3 ,c_tag=0.5, groups=3, dila=1):
        super(CSA_part, self).__init__()
        self.__dw = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=kernel_size, stride=1, pad=(kernel_size-1)//2, groups=filters_in, dila=dila, norm="bn")
        self.__pw1 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.groups = groups

    def channel_shuffle(self, features):
        batchsize, num_channels, height, width = features.data.size()
        assert (num_channels % self.groups == 0)
        channels_per_group = num_channels // self.groups
        features = features.view(batchsize, self.groups, channels_per_group, height, width)# reshape
        features = torch.transpose(features, 1, 2).contiguous()
        features = features.view(batchsize, -1, height, width)# flatten
        return features
    def forward(self, x):
        right = self.__dw(x)
        right = self.__pw1(right)
        cat = torch.cat((x, right), 1)
        out = self.channel_shuffle(cat)
        return out

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
    def __init__(self, fileters_in, model_size=1):
        super(Neck, self).__init__()
        fi_0, fi_1, fi_2 = fileters_in
        self.__fo = (cfg.DATA["NUM"]+5+5)
        fm_0 = int(1024*model_size)
        fm_1 = fm_0 // 2
        fm_2 = fm_0 // 4

        self.__dcn2_1 = Deformable_Convolutional(fi_2, fi_2, kernel_size=3, stride=2, pad=1, groups=1)
        self.__routdcn2_1 = Route()

        self.__dcn1_0 = Deformable_Convolutional(fi_1+fi_2, fi_1, kernel_size=3, stride=2, pad=1, groups=1)

        self.__routdcn1_0 = Route()
        # large
        self.__conv_set_0 = nn.Sequential(
            Convolutional(filters_in=fi_0 + fi_1, filters_out=fm_0, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            DRF(filters_in=fm_0, filters_out=fm_0, groups=8, dila_l=4, dila_r=6),#, dila_l=4, dila_r=6
            CSA_part(filters_in=fm_0//2, filters_out=fm_0, groups=8),
        )
        self.__conv0_0 = CSA(filters_in=fm_0, filters_out=fm_0, groups=4)
        self.__conv0_1 = Convolutional(filters_in=fm_0, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        self.__conv0up1 = nn.Conv2d(fm_0, fm_1, kernel_size=1, stride=1, padding=0)
        self.__upsample0_1 = Upsample(scale_factor=2)

        # medium
        self.__pw1 = Convolutional(filters_in=fi_2+fi_1, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")#, groups=fi_2+fi_1
        self.__shuffle10 = CSA(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__route0_1 = Route()
        self.__conv_set_1 = nn.Sequential(
            Convolutional(filters_in=fm_1*2, filters_out=fm_1, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            DRF(filters_in=fm_1, filters_out=fm_1, groups=4, dila_l=2, dila_r=3),#, dila_l=2, dila_r=3
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            CSA_part(filters_in=fm_1//2, filters_out=fm_1, groups=4),
        )
        self.__conv1_0 = CSA(filters_in=fm_1, filters_out=fm_1, groups=4)
        self.__conv1_1 = Convolutional(filters_in=fm_1, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        self.__conv1up2 = nn.Conv2d(fm_1, fm_2, kernel_size=1, stride=1, padding=0)
        self.__upsample1_2 = Upsample(scale_factor=2)


        # small
        self.__pw2 = Convolutional(filters_in=fi_2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky")
        self.__shuffle20 = CSA(filters_in=fm_2, filters_out=fm_2, groups=4)
        self.__route1_2 = Route()
        self.__conv_set_2 = nn.Sequential(
            Convolutional(filters_in=fm_2*2, filters_out=fm_2, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            CSA(filters_in=fm_2, filters_out=fm_2, groups=4),
            LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0., stop_value=0.1, nr_steps=5),
            CSA(filters_in=fm_2, filters_out=fm_2, groups=4),
        )
        self.__conv2_0 = CSA(filters_in=fm_2, filters_out=fm_2, groups=4)
        self.__conv2_1 = Convolutional(filters_in=fm_2, filters_out=self.__fo, kernel_size=1, stride=1, pad=0)

        self.__initialize_weights()


    def __initialize_weights(self):
        print("**" * 10, "Initing FPN_YOLOV3 weights", "**" * 10)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
                print("initing {}".format(m))

            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

    def forward(self, x0, x1, x2):

        dcn2_1 = self.__dcn2_1(x2)
        routdcn2_1 = self.__routdcn2_1(x1, dcn2_1)

        dcn1_0  = self.__dcn1_0(routdcn2_1)
        routdcn1_0 = self.__routdcn1_0(x0, dcn1_0)

        # large
        conv_set_0 = self.__conv_set_0(routdcn1_0)
        conv0up1 = self.__conv0up1(conv_set_0)
        upsample0_1 = self.__upsample0_1(conv0up1)

        # medium
        pw1 = self.__pw1(routdcn2_1)
        shuffle10 = self.__shuffle10(pw1)
        route0_1 = self.__route0_1(shuffle10,upsample0_1)
        conv_set_1 = self.__conv_set_1(route0_1)

        conv1up2 = self.__conv1up2(conv_set_1)
        upsample1_2 = self.__upsample1_2(conv1up2)

        # small
        pw2 = self.__pw2(x2)
        shuffle20 = self.__shuffle20(pw2)
        route1_2 = self.__route1_2(shuffle20, upsample1_2)
        conv_set_2 = self.__conv_set_2(route1_2)

        out0 = self.__conv0_0(conv_set_0)
        out0 = self.__conv0_1(out0)

        out1 = self.__conv1_0(conv_set_1)
        out1 = self.__conv1_1(out1)

        out2 = self.__conv2_0(conv_set_2)
        out2 = self.__conv2_1(out2)

        return out2, out1, out0  # small, medium, large