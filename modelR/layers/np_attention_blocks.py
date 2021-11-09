import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from ..layers.convolutions import Convolutional

def sobel_kernel(channel_in, channel_out, theta):
    sobel_kernel0 = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype='float32')
    sobel_kernel0 = sobel_kernel0.reshape((1, 1, 3, 3))
    sobel_kernel0 = Variable(torch.from_numpy(sobel_kernel0))
    sobel_kernel0 = sobel_kernel0.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel0 = sobel_kernel0.cuda()*theta.view(-1, 1, 1, 1).cuda()

    sobel_kernel45 = np.array([[2, 1, 0], [1, 0, -1], [0, -1, -2]], dtype='float32')
    sobel_kernel45 = sobel_kernel45.reshape((1, 1, 3, 3))
    sobel_kernel45 = Variable(torch.from_numpy(sobel_kernel45))
    sobel_kernel45 = sobel_kernel45.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel45 = sobel_kernel45.cuda()*theta.view(-1, 1, 1, 1).cuda()

    sobel_kernel90 = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype='float32')
    sobel_kernel90 = sobel_kernel90.reshape((1, 1, 3, 3))
    sobel_kernel90 = Variable(torch.from_numpy(sobel_kernel90))
    sobel_kernel90 = sobel_kernel90.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel90 = sobel_kernel90.cuda()*theta.view(-1, 1, 1, 1).cuda()

    sobel_kernel135 = np.array([[0, -1, -2], [1, 0, -1], [2, 1, 0]], dtype='float32')
    sobel_kernel135 = sobel_kernel135.reshape((1, 1, 3, 3))
    sobel_kernel135 = Variable(torch.from_numpy(sobel_kernel135))
    sobel_kernel135 = sobel_kernel135.repeat(channel_out, channel_in, 1, 1).float()
    sobel_kernel135 = sobel_kernel135.cuda()*theta.view(-1, 1, 1, 1).cuda()

    return sobel_kernel0, sobel_kernel45, sobel_kernel90, sobel_kernel135

class Sobel_conv(nn.Module):
    def __init__(self, channel_in, channel_out, alpha=0.5, sigma=4, stride=1, padding=1):
        super(Sobel_conv, self).__init__()
        self.channel_in = channel_in
        self.channel_out = channel_out
        self.stride = stride
        self.padding = padding
        self.sigma = sigma
        self.alpha = alpha
        self.__conv_weight = Convolutional(channel_out * 4, 4, kernel_size=1, stride=1, pad=0, norm='bn', activate='leaky')
        self.theta = nn.Parameter(torch.sigmoid(torch.randn(channel_out) * 1.0) + self.alpha, requires_grad=True)

    def forward(self, x):
        # [channel_out, channel_in, kernel, kernel]
        kernel0, kernel45, kernel90, kernel135 = sobel_kernel(self.channel_in, self.channel_out, self.theta)
        kernel0 = kernel0.float()
        kernel45 = kernel45.float()
        kernel90 = kernel90.float()
        kernel135 = kernel135.float()

        out0 = F.conv2d(x, kernel0, stride=self.stride, padding=self.padding)
        out45 = F.conv2d(x, kernel45, stride=self.stride, padding=self.padding)
        out90 = F.conv2d(x, kernel90, stride=self.stride, padding=self.padding)
        out135 = F.conv2d(x, kernel135, stride=self.stride, padding=self.padding)

        out_cat = torch.cat((out0, out45, out90, out135),1)
        out_cat_conv = self.__conv_weight(out_cat)
        out_weight = F.softmax(out_cat_conv, dim=1)

        out = torch.abs(out0)* out_weight[:,0:1,:,:] + torch.abs(out45)*out_weight[:,1:2,:,:]\
              + torch.abs(out90)*out_weight[:,2:3,:,:] + torch.abs(out135)*out_weight[:,3:,:,:]
        out = (out * self.sigma)
        return out

class Sobel_Edge_Block(nn.Module):
    def __init__(self, channel_in, alpha=0.5, sigma=4):
        super(Sobel_Edge_Block, self).__init__()
        self.__down0 = nn.AvgPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.__sobelconv = Sobel_conv(channel_in, channel_in, alpha, sigma)
        self.__down1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.__conv0 = Convolutional(channel_in, 2, kernel_size=1, stride=1, pad=0, norm='bn', activate='Mish')

    def forward(self, x):
        x_down0 = self.__down0(x)
        x_sobel = self.__sobelconv(x_down0)
        x_down1 = self.__down1(self.__down1(x_sobel))
        x_conv0 = self.__conv0(x_down1)
        return x_conv0

class NPAttention(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(NPAttention, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        self.t_mask = nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes//2, kernel_size=1, stride=1, bias=False)
        self.p1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        # conv g
        self.g = nn.Conv2d(inplanes, planes//2, kernel_size=1, stride=1, bias=False)
        self.g1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.g2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

    def kernel(self, t, p, g, b, c, h, w):
        #The linear kernel (dot production)
        t = t.view(b, c, h * w)
        t = self.softmax(t)
        t = t.view(b, 1, c * h * w)
        p = p.view(b, 1, c * h * w)
        g = g.view(b, c * h * w, 1)
        att = torch.bmm(p, g)
        if self.use_scale:
            att = att.div((c*h*w)**0.5)
        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)
        return x

    def forward(self, x, mask):
        residual = x
        t = self.t(x)
        t_mask = self.t_mask(mask)
        t = t*t_mask+t
        p = self.p(x)
        p = torch.cat((p,self.p1(p),self.p2(p)),1)
        g = self.g(x)
        g = torch.cat((g,self.g1(g),self.g2(g)),1)
        b, c, h, w = t.size()
        if self.groups and self.groups > 1:
            _c = int(c / self.groups)
            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)
            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h, w)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h, w)
        x = self.z(x)
        xout = self.gn(x)
        out = xout + residual
        return out


class NPAttention1(nn.Module):
    def __init__(self, inplanes, planes, use_scale=False, groups=None):
        self.use_scale = use_scale
        self.groups = groups

        super(NPAttention1, self).__init__()
        # conv theta
        self.t = nn.Conv2d(inplanes, planes, kernel_size=1, stride=1, bias=False)
        self.softmax = nn.Softmax(dim=2)
        #self.t_mask = nn.Conv2d(2, 1, kernel_size=1, stride=1, bias=False)
        # conv phi
        self.p = nn.Conv2d(inplanes, planes//2, kernel_size=1, stride=1, bias=False)
        self.p1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.p2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        # conv g
        self.g = nn.Conv2d(inplanes, planes//2, kernel_size=1, stride=1, bias=False)
        self.g1 = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.g2 = nn.MaxPool2d(kernel_size=5, stride=1, padding=2)
        # conv z
        self.z = nn.Conv2d(planes, inplanes, kernel_size=1, stride=1, groups=self.groups, bias=False)
        self.gn = nn.GroupNorm(num_groups=self.groups, num_channels=inplanes)

    def kernel(self, t, p, g, b, c, h, w):
        #The linear kernel (dot production)
        t = t.view(b, c, h * w)
        t = self.softmax(t)
        t = t.contiguous().view(b, 1, c * h * w)
        p = p.contiguous().view(b, 1, c * h * w)
        g = g.contiguous().view(b, c * h * w, 1)
        att = torch.bmm(p, g)
        if self.use_scale:
            att = att.div((c*h*w)**0.5)
        x = torch.bmm(att, t)
        x = x.view(b, c, h, w)
        return x

    def forward(self, x):
        residual = x
        t = self.t(x)
        #t_mask = self.t_mask(mask)
        #t = t*t_mask+t
        p = self.p(x)
        p = torch.cat((p,self.p1(p),self.p2(p)),1)
        g = self.g(x)
        g = torch.cat((g,self.g1(g),self.g2(g)),1)
        b, c, h, w = t.size()
        if self.groups and self.groups > 1:
            _c = int(c / self.groups)
            ts = torch.split(t, split_size_or_sections=_c, dim=1)
            ps = torch.split(p, split_size_or_sections=_c, dim=1)
            gs = torch.split(g, split_size_or_sections=_c, dim=1)
            _t_sequences = []
            for i in range(self.groups):
                _x = self.kernel(ts[i], ps[i], gs[i], b, _c, h, w)
                _t_sequences.append(_x)
            x = torch.cat(_t_sequences, dim=1)
        else:
            x = self.kernel(t, p, g, b, c, h, w)
        x = self.z(x)
        xout = self.gn(x)
        out = xout + residual
        return out
