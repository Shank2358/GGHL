from .activations import *
#from dcn_v2 import DCN
#from modelR.layers.deform_conv_v2 import DeformConv2d
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from dcn_v2 import DCN

norm_name = {"bn": nn.BatchNorm2d}#
activate_name = {
    "relu": nn.ReLU,
    "leaky": nn.LeakyReLU,
    "relu6": nn.ReLU6,
    "Mish": Mish,
    "Swish": Swish,
    "MEMish": MemoryEfficientMish,
    "MESwish": MemoryEfficientSwish,
    "FReLu": FReLU
}

class Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, groups=1, dila=1, norm=None, activate=None):
        super(Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = nn.Conv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                stride=stride, padding=pad, bias=not norm, groups=groups, dilation=dila)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        x = self.__conv(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x



class Deformable_Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride, pad, groups=1, norm=None, activate=None):
        super(Deformable_Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__dcn = DCN(filters_in, filters_out, kernel_size=kernel_size, stride=stride, padding=pad, deformable_groups=groups)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        x = self.__dcn(x)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x

class route_func(nn.Module):
    r"""CondConv: Conditionally Parameterized Convolutions for Efficient Inference
    https://papers.nips.cc/paper/8412-condconv-conditionally-parameterized-convolutions-for-efficient-inference.pdf
    Args:
        c_in (int): Number of channels in the input image
        num_experts (int): Number of experts for mixture. Default: 1
    """

    def __init__(self, c_in, num_experts):
        super(route_func, self).__init__()
        self.avgpool = nn.AdaptiveAvgPool2d(output_size=1)
        self.fc = nn.Linear(c_in, num_experts)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        x = self.sigmoid(x)
        return x

class CondConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size,
                 stride=1, padding=0, dilation=1, groups=1, bias=True,
                 num_experts=1):
        super(CondConv2d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.groups = groups
        self.num_experts = num_experts

        self.weight = nn.Parameter(torch.Tensor(num_experts, out_channels, in_channels // groups, kernel_size, kernel_size))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(num_experts, out_channels))
        else:
            self.register_parameter('bias', None)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, routing_weight):
        b, c_in, h, w = x.size()
        k, c_out, c_in, kh, kw = self.weight.size()
        x = x.contiguous().view(1, -1, h, w)
        weight = self.weight.contiguous().view(k, -1)

        combined_weight = torch.mm(routing_weight, weight).view(-1, c_in, kh, kw)

        if self.bias is not None:
            combined_bias = torch.mm(routing_weight, self.bias).view(-1)
            output = F.conv2d(
                x, weight=combined_weight, bias=combined_bias, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)
        else:
            output = F.conv2d(
                x, weight=combined_weight, bias=None, stride=self.stride, padding=self.padding,
                dilation=self.dilation, groups=self.groups * b)

        output = output.view(b, c_out, output.size(-2), output.size(-1))
        return output

class Cond_Convolutional(nn.Module):
    def __init__(self, filters_in, filters_out, kernel_size, stride=1, pad=0, dila=1, groups=1, bias=True, num_experts=1, norm=None, activate=None):

        super(Cond_Convolutional, self).__init__()
        self.norm = norm
        self.activate = activate
        self.__conv = CondConv2d(in_channels=filters_in, out_channels=filters_out, kernel_size=kernel_size,
                                 stride=stride, padding=pad, dilation=dila, groups=groups, bias=bias, num_experts=num_experts)
        self.__routef = route_func(filters_in, num_experts)
        if norm:
            assert norm in norm_name.keys()
            if norm == "bn":
                self.__norm = norm_name[norm](num_features=filters_out)
        if activate:
            assert activate in activate_name.keys()
            if activate == "leaky":
                self.__activate = activate_name[activate](negative_slope=0.1, inplace=True)
            if activate == "relu":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "relu6":
                self.__activate = activate_name[activate](inplace=True)
            if activate == "Mish":
                self.__activate = Mish()
            if activate == "Swish":
                self.__activate = Swish()
            if activate == "MEMish":
                self.__activate = MemoryEfficientMish()
            if activate == "MESwish":
                self.__activate = MemoryEfficientSwish()
            if activate == "FReLu":
                self.__activate = FReLU()

    def forward(self, x):
        routef = self.__routef(x)
        x = self.__conv(x,routef)
        if self.norm:
            x = self.__norm(x)
        if self.activate:
            x = self.__activate(x)
        return x

