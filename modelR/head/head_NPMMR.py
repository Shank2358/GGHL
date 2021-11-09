import torch.nn as nn
import torch
import torch.nn.functional as F
from lib.DCNv2.DCN import DCNv2

class hsigmoid(nn.Module):
    def forward(self, x):
        out = F.relu6(x + 3, inplace=True) / 6
        return out

class MTR_Head1(nn.Module):
    def __init__(self, filters_in, fo_class, temp=False):
        super(MTR_Head1, self).__init__()
        self.fo_class = fo_class
        self.temp = temp

        self.__conv_conf = nn.Conv2d(in_channels=filters_in, out_channels=2, kernel_size=1, stride=1,padding=0)
        self.__conv_offset_mask = nn.Conv2d(in_channels=filters_in, out_channels=3 * 9, kernel_size=1, stride=1,padding=0, bias=True)

        self.__dconv_loc = DCNv2(filters_in, filters_in, kernel_size=3, stride=1, padding=1)
        self.__bnloc = nn.BatchNorm2d(filters_in)
        self.__reluloc = nn.LeakyReLU(inplace=True)
        self.__dconv_locx = nn.Conv2d(filters_in, 8, kernel_size=1, stride=1, padding=0)

        self.__dconv_cla = DCNv2(filters_in, filters_in, kernel_size=3, stride=1, padding=1)
        self.__bncla = nn.BatchNorm2d(filters_in)
        self.__relucla = nn.LeakyReLU(inplace=True)
        self.__dconv_clax = nn.Conv2d(filters_in, self.fo_class, kernel_size=1, stride=1, padding=0)

        self.init_offset()

    def init_offset(self):
        self.__conv_offset_mask.weight.data.zero_()
        self.__conv_offset_mask.bias.data.zero_()

    def forward(self, x):
        out_conf = self.__conv_conf(x)

        out_offset_mask = self.__conv_offset_mask(x)
        o1, o2, mask = torch.chunk(out_offset_mask, 3, dim=1)
        offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(mask)

        out_loc = self.__dconv_locx(self.__reluloc(self.__bnloc(self.__dconv_loc(x, offset, mask))))
        out_cla = self.__dconv_clax(self.__relucla(self.__bncla(self.__dconv_cla(x, offset, mask))))

        out_loc1 = out_loc.view(x.shape[0], 8, x.shape[2], x.shape[3]).cuda()
        out_conf1 = out_conf.view(x.shape[0], 2, x.shape[2], x.shape[3]).cuda()#######
        out_cla1 = out_cla.view(x.shape[0], self.fo_class, x.shape[2], x.shape[3]).cuda()
        out = torch.cat((out_loc1, out_conf1, out_cla1), 1).cuda()
        return out

class MTR_Head2(nn.Module):
    def __init__(self, nC, stride):
        super(MTR_Head2, self).__init__()
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        p = p.permute(0, 2, 3, 1)
        p_de = self.__decode(p.clone())
        return (p, p_de)
    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        conv_raw_l1234 = p[:, :, :, 0:4]
        conv_raw_s = p[:, :, :, 4:8]
        conv_raw_r = p[:, :, :, 8:9]
        conv_raw_conf = p[:, :, :, 9:10]
        conv_raw_prob = p[:, :, :, 10:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(device)

        pred_l1234 = torch.exp(conv_raw_l1234) * stride
        pred_xmin = grid_xy[:, :, :, 0:1] * stride + (stride) / 2 - pred_l1234[:, :, :, 3:4]
        pred_ymin = grid_xy[:, :, :, 1:2] * stride + (stride) / 2 - pred_l1234[:, :, :, 0:1]
        pred_xmax = grid_xy[:, :, :, 0:1] * stride + (stride) / 2 + pred_l1234[:, :, :, 1:2]
        pred_ymax = grid_xy[:, :, :, 1:2] * stride + (stride) / 2 + pred_l1234[:, :, :, 2:3]

        pred_w = (pred_l1234[:, :, :, 1:2] + pred_l1234[:, :, :, 3:4])
        pred_h = (pred_l1234[:, :, :, 0:1] + pred_l1234[:, :, :, 2:3])
        pred_x = (pred_xmax + pred_xmin) / 2
        pred_y = (pred_ymax + pred_ymin) / 2
        pred_xywh = torch.cat([pred_x, pred_y, pred_w, pred_h], dim=-1)
        pred_s = (torch.clamp(torch.sigmoid(conv_raw_s), 0.01, 1) - 0.01) / (1 - 0.01)
        pred_r = F.relu6(conv_raw_r + 3, inplace=True) / 6
        maskr = pred_r
        zero = torch.zeros_like(maskr)
        one = torch.ones_like(maskr)
        maskr = torch.where(maskr > 0.9, zero, one)  # 0.8
        pred_s[:, :, :, 0:1] = pred_s[:, :, :, 0:1] * maskr
        pred_s[:, :, :, 1:2] = pred_s[:, :, :, 1:2] * maskr
        pred_s[:, :, :, 2:3] = pred_s[:, :, :, 2:3] * maskr
        pred_s[:, :, :, 3:4] = pred_s[:, :, :, 3:4] * maskr
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)

        pred_bbox = torch.cat([pred_xywh, pred_s, pred_r, pred_l1234, pred_conf, pred_prob], dim=-1)
        out = pred_bbox.view(-1, 4 + 5 + 4 + self.__nC + 1) if not self.training else pred_bbox
        return out
