from dropblock import DropBlock2D, LinearScheduler
from ..layers.convolutions import *

class MSR_L(nn.Module):
    def __init__(self, filters_in):
        super(MSR_L, self).__init__()
        self.__dw0 = Convolutional(filters_in=filters_in, filters_out=filters_in*2, kernel_size=3,
                                   stride=1, pad=1, norm="bn", activate="leaky")
        self.__pw0 = Convolutional(filters_in=filters_in*2, filters_out=filters_in, kernel_size=1,
                                   stride=1, pad=0, norm="bn", activate="leaky")
        self.__dw1 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                   stride=1, pad=2, dila=2, norm="bn", activate="leaky")
        self.__dw2 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                   stride=1, pad=4, dila=4, norm="bn", activate="leaky")
        self.__dw3 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                   stride=1, pad=6, dila=6, norm="bn", activate="leaky")
        self.__pw1 = Convolutional(filters_in=filters_in*4, filters_out=filters_in, kernel_size=1,
                                   stride=1, pad=0, norm="bn", activate="Mish")
        self.__drop = LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0.,
                                      stop_value=0.1, nr_steps=5)

    def forward(self, x):
        dw0 = self.__dw0(x)
        dw0 = self.__drop(dw0)
        pw0 = self.__pw0(dw0)
        dw1 = self.__dw1(pw0)
        dw2 = self.__dw2(pw0)+dw1
        dw3 = self.__dw3(pw0)+dw2
        cat = torch.cat((pw0, dw1, dw2, dw3),1)
        pw1 = self.__pw1(cat)
        return pw1

class MSR_M(nn.Module):
    def __init__(self, filters_in):
        super(MSR_M, self).__init__()
        self.__dw0 = Convolutional(filters_in=filters_in, filters_out=filters_in*2, kernel_size=3,
                                   stride=1, pad=1, norm="bn", activate="leaky")
        self.__pw0 = Convolutional(filters_in=filters_in*2, filters_out=filters_in, kernel_size=1,
                                   stride=1, pad=0, norm="bn", activate="leaky")
        self.__dw1 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                   stride=1, pad=2, dila=2, norm="bn", activate="leaky")
        self.__dw2 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                   stride=1, pad=4, dila=4, norm="bn", activate="leaky")
        self.__pw1 = Convolutional(filters_in=filters_in*2, filters_out=filters_in, kernel_size=1,
                                   stride=1, pad=0, norm="bn", activate="Mish")
        self.__drop = LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0.,
                                      stop_value=0.1, nr_steps=5)

    def forward(self, x):
        dw0 = self.__dw0(x)
        dw0 = self.__drop(dw0)
        pw0 = self.__pw0(dw0)
        dw1 = self.__dw1(pw0)
        dw2 = self.__dw2(pw0)+dw1
        cat = torch.cat((dw1, dw2),1)
        pw1 = self.__pw1(cat)
        return pw1

class MSR_S(nn.Module):
    def __init__(self, filters_in):
        super(MSR_S, self).__init__()
        self.__dw0 = Convolutional(filters_in=filters_in, filters_out=filters_in*2, kernel_size=3,
                                   stride=1, pad=1, norm="bn", activate="leaky")
        self.__pw0 = Convolutional(filters_in=filters_in*2, filters_out=filters_in, kernel_size=1,
                                   stride=1, pad=0, norm="bn", activate="leaky")
        self.__dw1 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                   stride=1, pad=1, dila=1, norm="bn", activate="leaky")
        self.__dw2 = Convolutional(filters_in=filters_in, filters_out=filters_in, kernel_size=3,
                                   stride=1, pad=2, dila=2, norm="bn", activate="leaky")
        self.__pw1 = Convolutional(filters_in=filters_in*2, filters_out=filters_in, kernel_size=1,
                                   stride=1, pad=0, norm="bn", activate="Mish")
        self.__drop = LinearScheduler(DropBlock2D(block_size=3, drop_prob=0.1), start_value=0.,
                                      stop_value=0.1, nr_steps=5)

    def forward(self, x):
        dw0 = self.__dw0(x)
        dw0 = self.__drop(dw0)
        pw0 = self.__pw0(dw0)
        dw1 = self.__dw1(pw0)
        dw2 = self.__dw2(pw0)+dw1
        cat = torch.cat((dw1, dw2),1)
        pw1 = self.__pw1(cat)
        return pw1
