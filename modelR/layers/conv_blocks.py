import torch
import torch.nn as nn
from ..layers.convolutions import Convolutional

class Residual_block(nn.Module):
    def __init__(self, filters_in, filters_out, filters_medium, norm="bn", activate="leaky"):
        super(Residual_block, self).__init__()
        self.__conv1 = Convolutional(filters_in=filters_in, filters_out=filters_medium, kernel_size=1,
                                     stride=1, pad=0, norm=norm, activate=activate)
        self.__conv2 = Convolutional(filters_in=filters_medium, filters_out=filters_out, kernel_size=3,
                                     stride=1, pad=1, norm=norm, activate=activate)

    def forward(self, x):
        r = self.__conv1(x)
        r = self.__conv2(r)
        out = x + r
        return out

class CSP_stage(nn.Module):
    def __init__(self, filters_in, n=1, activate="Swish"):
        super(CSP_stage, self).__init__()
        c_ = filters_in // 2  # hidden channels
        self.conv1 = Convolutional(filters_in=filters_in, filters_out=c_, kernel_size=1, stride=1, pad=0, norm="bn", activate=activate)
        self.conv2 = Convolutional(filters_in=filters_in, filters_out=c_, kernel_size=1, stride=1, pad=0, norm="bn", activate=activate)
        self.res_blocks = nn.Sequential(*[Residual_block(filters_in=c_, filters_out=c_, filters_medium=c_, norm="bn", activate=activate) for _ in range(n)])
        self.conv3 = Convolutional(filters_in=2 * c_, filters_out=filters_in, kernel_size=1, stride=1, pad=0, norm="bn", activate=activate)

    def forward(self, x):
        y1 = self.conv1(x)
        y2 = self.conv2(x)
        y2 = self.res_blocks(y2)
        return self.conv3(torch.cat([y2, y1], dim=1))

class Residual_block_CSP(nn.Module):
    def __init__(self, filters_in):
        super(Residual_block_CSP, self).__init__()
        self.__conv1 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=1,
                                     stride=1, pad=0, norm="bn", activate="leaky")
        self.__conv2 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                     stride=1, pad=1, norm="bn", activate="leaky")

    def forward(self, x):
        r = self.__conv1(x)
        r = self.__conv2(r)
        out = x + r
        return out


class InvertedResidual_block(nn.Module):
    def __init__(self, inp, oup, stride, expand_ratio):
        super(InvertedResidual_block, self).__init__()
        self.__stride = stride
        hidden_dim = int(inp * expand_ratio)
        self.use_res_connect = self.__stride == 1 and inp==oup
        if expand_ratio==1:
            self.__conv = nn.Sequential(
                Convolutional(filters_in=hidden_dim, filters_out=hidden_dim, kernel_size=3,
                              stride=self.__stride, pad=1, groups=hidden_dim, norm="bn", activate="relu6"),
                Convolutional(filters_in=hidden_dim, filters_out=oup, kernel_size=1,
                              stride=1, pad=0, norm="bn")
            )
        else:
            self.__conv = nn.Sequential(
                Convolutional(filters_in=inp, filters_out=hidden_dim, kernel_size=1,
                              stride=1, pad=0, norm="bn", activate="relu6"),
                Convolutional(filters_in=hidden_dim, filters_out=hidden_dim, kernel_size=3,
                              stride=self.__stride, pad=1, groups=hidden_dim, norm="bn", activate="relu6"),
                Convolutional(filters_in=hidden_dim, filters_out=oup, kernel_size=1,
                              stride=1, pad=0, norm="bn")
            )

    def forward(self, x):
        if self.use_res_connect:
            return x + self.__conv(x)
        else:
            return self.__conv(x)






