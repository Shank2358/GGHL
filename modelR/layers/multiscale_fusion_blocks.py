import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.convolutions import Convolutional, Deformable_Convolutional

class SPP(nn.Module):
    def __init__(self, depth=512):
        super(SPP,self).__init__()
        self.__maxpool5 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        self.__maxpool9 = nn.MaxPool2d(kernel_size=9, stride=1, padding=4)
        self.__maxpool13 = nn.MaxPool2d(kernel_size=13, stride=1, padding=6)
        self.__outconv = nn.Conv2d(depth * 4, depth, 1, 1)

    def forward(self, x):
        maxpool5 = self.__maxpool5(x)
        maxpool9 = self.__maxpool9(x)
        maxpool13 = self.__maxpool13(x)
        cat_maxpool = torch.cat([x, maxpool5, maxpool9, maxpool13], dim=1)
        SPP = self.__outconv(cat_maxpool)
        return SPP

class ASPP(nn.Module):
    def __init__(self, in_channel=1280, depth=512):
        super(ASPP,self).__init__()
        self.__dilaconv1 = nn.Conv2d(in_channel, depth, 1, 1)
        self.__dilaconv5 = nn.Conv2d(in_channel, depth, 3, 1, padding=2, dilation=2)
        self.__dilaconv9 = nn.Conv2d(in_channel, depth, 3, 1, padding=4, dilation=4)
        self.__dilaconv13 = nn.Conv2d(in_channel, depth, 3, 1, padding=6, dilation=6)
        self.__outconv = nn.Conv2d(depth * 4, depth, 1, 1)

    def forward(self, x):
        dilaconv1 = self.__dilaconv1(x)
        dilaconv5 = self.__dilaconv5(x)
        dilaconv9 = self.__dilaconv9(x)
        dilaconv13 = self.__dilaconv13(x)
        cat_dilaconv = torch.cat([dilaconv1, dilaconv5, dilaconv9, dilaconv13], dim=1)
        ASPP = self.__outconv(cat_dilaconv)
        return ASPP

class ASFF(nn.Module):
    def __init__(self, level, vis=False):
        super(ASFF, self).__init__()
        self.level = level
        self.dim = [512,256,128]
        self.inter_dim = self.dim[self.level]
        if level == 0:
            self.stride_level_1 = Convolutional(256, self.inter_dim, 3, 2, pad=1, norm='bn', activate='relu6')
            self.stride_level_2 = Convolutional(128, self.inter_dim, 3, 2,  pad=1, norm='bn', activate='relu6')
            self.expand = Convolutional(self.inter_dim, 1024, 3, 1,  pad=1, norm='bn', activate='relu6')
        elif level == 1:
            self.compress_level_0 = Convolutional(512, self.inter_dim, 1, 1,  pad=0, norm='bn', activate='relu6')
            self.stride_level_2 = Convolutional(128, self.inter_dim, 3, 2, pad=1, norm='bn', activate='relu6')
            self.expand = Convolutional(self.inter_dim, 512, 3, 1, pad=1, norm='bn', activate='relu6')
        elif level == 2:
            self.compress_level_0 = Convolutional(512, self.inter_dim,  1, 1, pad=0, norm='bn', activate='relu6')
            self.compress_level_1 = Convolutional(256, self.inter_dim,  1, 1, pad=0, norm='bn', activate='relu6')
            self.expand = Convolutional(self.inter_dim, 256, 3, 1, pad=1, norm='bn', activate='relu6')
        compress_c = 16
        self.weight_level_0 = Convolutional(self.inter_dim, compress_c, 1, 1, pad=0, norm='bn', activate='relu6')
        self.weight_level_1 = Convolutional(self.inter_dim, compress_c, 1, 1, pad=0, norm='bn', activate='relu6')
        self.weight_level_2 = Convolutional(self.inter_dim, compress_c, 1, 1, pad=0, norm='bn', activate='relu6')
        self.weight_levels = nn.Conv2d(compress_c * 3, 3, kernel_size=1, stride=1, padding=0)
        self.vis = vis

    def forward(self, x_level_0, x_level_1, x_level_2):
        if self.level == 0:
            level_0_resized = x_level_0
            level_1_resized = self.stride_level_1(x_level_1)
            level_2_downsampled_inter = F.max_pool2d(x_level_2, 3, stride=2, padding=1)
            level_2_resized = self.stride_level_2(level_2_downsampled_inter)
        elif self.level == 1:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=2, mode='nearest')
            level_1_resized = x_level_1
            level_2_resized = self.stride_level_2(x_level_2)
        elif self.level == 2:
            level_0_compressed = self.compress_level_0(x_level_0)
            level_0_resized = F.interpolate(level_0_compressed, scale_factor=4, mode='nearest')
            level_1_compressed = self.compress_level_1(x_level_1)
            level_1_resized = F.interpolate(level_1_compressed, scale_factor=2, mode='nearest')
            level_2_resized = x_level_2

        level_0_weight_v = self.weight_level_0(level_0_resized)
        level_1_weight_v = self.weight_level_1(level_1_resized)
        level_2_weight_v = self.weight_level_2(level_2_resized)
        levels_weight_v = torch.cat((level_0_weight_v, level_1_weight_v, level_2_weight_v), 1)
        levels_weight = self.weight_levels(levels_weight_v)
        levels_weight = F.softmax(levels_weight, dim=1)

        fused_out_reduced = level_0_resized * levels_weight[:, 0:1, :, :] + \
                            level_1_resized * levels_weight[:, 1:2, :, :] + \
                            level_2_resized * levels_weight[:, 2:, :, :]

        out = self.expand(fused_out_reduced)

        if self.vis:
            return out, levels_weight, fused_out_reduced.sum(dim=1)
        else:
            return out

class FeatureAdaption(nn.Module):
    def __init__(self, in_ch, out_ch, n_anchors):
        super(FeatureAdaption, self).__init__()
        self.sep=False
        self.conv_offset = nn.Conv2d(in_channels=2*n_anchors,  out_channels=2*9*n_anchors, groups = n_anchors, kernel_size=1,stride=1,padding=0)
        self.dconv = Deformable_Convolutional(filters_in=in_ch, filters_out=out_ch, kernel_size=3, stride=1, pad=1, groups=n_anchors)

    def forward(self, input, wh_pred):
        wh_pred_new = wh_pred.detach()
        offset = self.conv_offset(wh_pred_new)
        out = self.dconv(input, offset)
        return out

class Features_Fusion(nn.Module):
    def __init__(self, in_channels, out_channels, r=16):
        super(Features_Fusion,self).__init__()
        self.out_channels = out_channels
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_fc1 = Convolutional(in_channels, in_channels // r, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.conv_fc2 = nn.Conv2d(in_channels // r, out_channels * 2, kernel_size=1, padding=0, bias=False)
        self.softmax = nn.Softmax(dim=2)


    def forward(self, x1, x2):
        batch_size = x1.size(0)
        x_mix = torch.add(x1,x2) # 逐元素相加生成 混合特征U
        x_avg = self.avg_pool(x_mix)
        x_fcout = self.conv_fc2(self.conv_fc1(x_avg)) # 先降维,后升维，结果中前一半通道值为a,后一半为b
        x_reshape = x_fcout.reshape(batch_size, self.out_channels, 2, -1)  # 调整形状，变为两个全连接层的值
        x_softmax = self.softmax(x_reshape)  # 使得两个全连接层对应位置进行softmax
        w1 = x_softmax[:, :, 0:1,:] #将tensor按照指定维度切分成2个tensor块
        w2 = x_softmax[:, :, 1:2,:]
        out = x1*w1 + x2*w2 # 两个加权后的特征 逐元素相加
        return out