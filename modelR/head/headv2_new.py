import torch.nn as nn
import torch
import torch.nn.functional as F
from dcn_v2_amp import DCNv2
from ..layers.convolutions import Convolutional
#from mmcv.ops import ModulatedDeformConv2d as DCNv2
#from mmcv.ops import DeformConv2dPack
# 使用方法与官方DCNv2一样，只不过deformable_groups参数名改为deform_groups即可，例如：
#dconv2 = DCN(in_channel, out_channel, kernel_size=(3, 3), stride=(2, 2), padding=1, deform_groups=2)

class Head1(nn.Module):  # 先预测第一个框和偏移量
    def __init__(self, filters_in, stride):
        super(Head1, self).__init__()
        self.__stride = stride
        self.__conv = Convolutional(filters_in=filters_in, filters_out=filters_in * 2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky")
        self.__conv_mask = nn.Conv2d(in_channels=filters_in * 2, out_channels=9 + 8, kernel_size=1, stride=1, padding=0, bias=True)  # , bias=True
        self.__loc1 = nn.Conv2d(in_channels=filters_in * 2, out_channels=10, kernel_size=1, stride=1, padding=0)

    def forward(self, input1):
        conv = self.__conv(input1)
        out1 = self.__loc1(conv).permute(0, 2, 3, 1).contiguous()
        conv_mask = self.__conv_mask(conv)
        #mask = torch.sigmoid(conv_mask)
        #o1, o2, mask = torch.chunk(conv_mask, 3, dim=1)
        #offset = torch.cat((o1, o2), dim=1)
        mask = torch.sigmoid(conv_mask[:, 0:9, :, :])
        offset_vertex = conv_mask[:, 9:, :, :].permute(0, 2, 3, 1).contiguous()
        out1_de, offsets = self.__decode(out1, offset_vertex)
        return out1, out1_de, offsets, mask

    def __decode(self, p, offset_vertex):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        conv_raw_l1234 = p[:, :, :, 0:4]
        conv_raw_s = p[:, :, :, 4:8]
        conv_raw_r = p[:, :, :, 8:9]
        conv_raw_conf = p[:, :, :, 9:10]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(device)###########################grid_xy嵌入进去，然后差分可变形卷积，其他特征变成attention

        pred_l1234 = conv_raw_l1234 ** 2 * stride
        pred_xmin = grid_xy[:, :, :, 0:1] * stride + (stride) / 2 - pred_l1234[:, :, :, 3:4]
        pred_ymin = grid_xy[:, :, :, 1:2] * stride + (stride) / 2 - pred_l1234[:, :, :, 0:1]
        pred_xmax = grid_xy[:, :, :, 0:1] * stride + (stride) / 2 + pred_l1234[:, :, :, 1:2]
        pred_ymax = grid_xy[:, :, :, 1:2] * stride + (stride) / 2 + pred_l1234[:, :, :, 2:3]

        pred_w = (pred_l1234[:, :, :, 1:2] + pred_l1234[:, :, :, 3:4])
        pred_h = (pred_l1234[:, :, :, 0:1] + pred_l1234[:, :, :, 2:3])
        pred_x = (pred_xmax + pred_xmin) / 2
        pred_y = (pred_ymax + pred_ymin) / 2
        pred_xywh = torch.cat([pred_x, pred_y, pred_w, pred_h], dim=-1)
        pred_s = (torch.clamp(torch.sigmoid(conv_raw_s), 0.01, 1) - 0.01) / (1 - 0.01)#torch.sigmoid(conv_raw_s)#
        pred_r = F.relu6(conv_raw_r + 3, inplace=True) / 6
        maskr = pred_r
        zero = torch.zeros_like(maskr)
        one = torch.ones_like(maskr)
        maskr = torch.where(maskr > 0.9, zero, one)  # 0.8
        pred_s[:, :, :, 0:1] = pred_s[:, :, :, 0:1] * maskr
        pred_s[:, :, :, 1:2] = pred_s[:, :, :, 1:2] * maskr
        pred_s[:, :, :, 2:3] = pred_s[:, :, :, 2:3] * maskr
        pred_s[:, :, :, 3:4] = pred_s[:, :, :, 3:4] * maskr
        ''''''
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_bbox = torch.cat([pred_xywh, pred_s, pred_r, pred_l1234, pred_conf], dim=-1)

        x1 = pred_xmin + pred_s[:, :, :, 0:1] * pred_w
        x3 = pred_xmax - pred_s[:, :, :, 2:3] * pred_w
        y2 = pred_ymin + pred_s[:, :, :, 1:2] * pred_h
        y4 = pred_ymax - pred_s[:, :, :, 3:4] * pred_h
        x_avg = (pred_xmin * 3 + pred_xmax * 3 + x1 + x3 + (pred_xmin+pred_xmax)/2) / 9
        y_avg = (pred_ymin * 3 + pred_ymax * 3 + y2 + y4 + (pred_ymin+pred_ymax)/2) / 9

        # offsets_x = torch.cat((v1x, p1x, v2x, p4x, cx, p2x, v3x, p3x, v4x), dim=-1)
        # offsets_y = torch.cat((v1y, p1y, v2y, p4y, cy, p2y, v3y, p3y, v4y), dim=-1)
        #
        #offsets_x = torch.cat((v1x, p4x, v4x, p1x, cx, p3x, v2x, p2x, v3x), dim=-1)
        #offsets_y = torch.cat((v1y, p4y, v4y, p1y, cy, p3y, v2y, p2y, v3y), dim=-1)
        '''
        offsets_x = torch.cat([pred_xmin / stride +1, x1 / stride, pred_xmax / stride -1,
                               pred_xmin / stride +1, x_avg / stride, pred_xmax / stride -1,
                               pred_xmin / stride +1, x3 / stride, pred_xmax / stride -1], dim=-1) \
                    - grid_xy[:, :, :, 0:1]

        offsets_y = torch.cat([pred_ymin / stride +1, pred_ymin / stride +1, pred_ymin / stride +1,
                               y4 / stride, y_avg / stride, y2 / stride,
                               pred_ymax / stride -1, pred_ymax / stride -1, pred_ymax / stride -1], dim=-1) \
                    - grid_xy[:, :, :, 1:2]
        offsets = torch.cat([offsets_y[:, :, :, 0:1], offsets_x[:, :, :, 0:1]], dim=-1).permute(0, 3, 1, 2)'''

        #off_y0 = pred_ymin / stride + 1 - grid_xy[:, :, :, 1:2]
        off_y1 = pred_ymin / stride + 1 - grid_xy[:, :, :, 1:2]
        #off_y2 = pred_ymin / stride + 1 - grid_xy[:, :, :, 1:2]
        off_y3 = y4 / stride - grid_xy[:, :, :, 1:2]
        off_y4 = y_avg / stride - grid_xy[:, :, :, 1:2]
        off_y5 = y2 / stride - grid_xy[:, :, :, 1:2]
        #off_y6 = pred_ymax / stride - 1 - grid_xy[:, :, :, 1:2]
        off_y7 = pred_ymax / stride - 1 - grid_xy[:, :, :, 1:2]
        #off_y8 = pred_ymax / stride - 1 - grid_xy[:, :, :, 1:2]

        #off_x0 = pred_xmin / stride + 1 - grid_xy[:, :, :, 0:1]
        off_x1 = x1 / stride - grid_xy[:, :, :, 0:1]
        #off_x2 = pred_xmax / stride - 1 - grid_xy[:, :, :, 0:1]

        off_x3 = pred_xmin / stride + 1 - grid_xy[:, :, :, 0:1]
        off_x4 = x_avg / stride - grid_xy[:, :, :, 0:1]
        off_x5 = pred_xmax / stride - 1 - grid_xy[:, :, :, 0:1]

        #off_x6 = pred_xmin / stride + 1 - grid_xy[:, :, :, 0:1]
        off_x7 = x3 / stride - grid_xy[:, :, :, 0:1]
        #off_x8 = pred_xmax / stride - 1 - grid_xy[:, :, :, 0:1]

        dc_off = torch.sigmoid(offset_vertex)
        dc_off_y0 = pred_ymin / stride + 1 - grid_xy[:, :, :, 1:2] + dc_off[:, :, :, 0:1] * pred_l1234[:, :, :, 0:1]
        dc_off_y2 = pred_ymin / stride + 1 - grid_xy[:, :, :, 1:2] + dc_off[:, :, :, 1:2] * pred_l1234[:, :, :, 0:1]
        dc_off_y6 = pred_ymax / stride - 1 - grid_xy[:, :, :, 1:2] - dc_off[:, :, :, 2:3] * pred_l1234[:, :, :, 2:3]
        dc_off_y8 = pred_ymax / stride - 1 - grid_xy[:, :, :, 1:2] - dc_off[:, :, :, 3:4] * pred_l1234[:, :, :, 2:3]

        dc_off_x0 = pred_xmin / stride + 1 - grid_xy[:, :, :, 0:1] + dc_off[:, :, :, 4:5] * pred_l1234[:, :, :, 3:4]
        dc_off_x2 = pred_xmax / stride - 1 - grid_xy[:, :, :, 0:1] - dc_off[:, :, :, 5:6] * pred_l1234[:, :, :, 1:2]
        dc_off_x6 = pred_xmin / stride + 1 - grid_xy[:, :, :, 0:1] + dc_off[:, :, :, 6:7] * pred_l1234[:, :, :, 3:4]
        dc_off_x8 = pred_xmax / stride - 1 - grid_xy[:, :, :, 0:1] - dc_off[:, :, :, 7:8] * pred_l1234[:, :, :, 1:2]

        offsets = torch.cat([dc_off_y0, dc_off_x0, off_y1, off_x1, dc_off_y2, dc_off_x2,
                             off_y3, off_x3, off_y4, off_x4, off_y5, off_x5,
                             dc_off_y6, dc_off_x6, off_y7, off_x7, dc_off_y8, dc_off_x8]
                            , dim=-1).permute(0, 3, 1, 2).contiguous()

        '''
        offsets_loc = torch.cat([off_y0, off_x0, off_y1, off_x1, off_y2, off_x2,
                             off_y3, off_x3, off_y4, off_x4, off_y5, off_x5,
                             off_y6, off_x6, off_y7, off_x7, off_y8, off_x8]
                            , dim=-1).permute(0, 3, 1, 2).contiguous()#'''

        return pred_bbox, offsets

class Head2(nn.Module):
    def __init__(self, filters_in, nC, stride):
        super(Head2, self).__init__()
        self.__nC = nC
        self.__stride = stride

        self.__dcn_loc = DCNv2(in_channels=filters_in, out_channels=filters_in * 2, kernel_size=3, padding=1, stride=1)
        self.__bn_loc = nn.BatchNorm2d(filters_in * 2)
        self.__relu_loc = nn.LeakyReLU(inplace=True)
        self.__conv_loc = nn.Sequential(
            Convolutional(filters_in=filters_in * 2, filters_out=filters_in, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=filters_in, filters_out=filters_in * 2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=filters_in * 2, out_channels=10, kernel_size=1, stride=1, padding=0)
        )

        self.__dcn_cls = DCNv2(in_channels=filters_in, out_channels=filters_in * 2, kernel_size=3, padding=1, stride=1)
        self.__bn_cls = nn.BatchNorm2d(filters_in * 2)
        self.__relu_cls = nn.LeakyReLU(inplace=True)
        self.__conv_cls = nn.Sequential(
            Convolutional(filters_in=filters_in * 2, filters_out=filters_in, kernel_size=1, stride=1, pad=0, norm="bn", activate="leaky"),
            Convolutional(filters_in=filters_in, filters_out=filters_in * 2, kernel_size=3, stride=1, pad=1, norm="bn", activate="leaky"),
            nn.Conv2d(in_channels=filters_in * 2, out_channels=self.__nC, kernel_size=1, stride=1, padding=0)
        )

    def forward(self, out1_de, loc, cls, offsets, mask):

        loc_dcn = self.__dcn_loc(loc, offsets, mask)
        loc_dcn = self.__relu_loc(self.__bn_loc(loc_dcn)) if self.training else self.__relu_loc(self.__bn_loc(loc_dcn)).float()
        #.float()

        conv_loc = self.__conv_loc(loc_dcn)

        cls_dcn = self.__dcn_cls(cls, offsets, mask)
        cls_dcn = self.__relu_cls(self.__bn_cls(cls_dcn)) if self.training else self.__relu_cls(self.__bn_cls(cls_dcn)).float()
        #.float()
        conv_cls = self.__conv_cls(cls_dcn)

        out2 = torch.cat((conv_loc, conv_cls), dim=1).permute(0, 2, 3, 1)
        out2_de = self.__decode(out1_de.detach(), out2.clone())
        return out2, out2_de

    def __decode(self, out1_de, out2):
        batch_size, output_size = out2.shape[:2]
        device = out2.device
        '''
        The offset tensor is like `[y0, x0, y1, x1, y2, x2, ..., y8, x8]`.
        The spatial arrangement is like:
        code:: text
        (x0, y0) (x1, y1) (x2, y2)
        (x3, y3) (x4, y4) (x5, y5)
        (x6, y6) (x7, y7) (x8, y8)
        '''
        conv_raw_l1234 = out2[:, :, :, 0:4]#######
        conv_raw_s = out2[:, :, :, 4:8]###############
        conv_raw_r = out2[:, :, :, 8:9]
        conv_raw_conf = out2[:, :, :, 9:10]
        conv_raw_prob = out2[:, :, :, 10:]

        out1_l = out1_de[:, :, :, 9:13]#########################
        out1_s = out1_de[:, :, :, 4:8]#####################
        #[pred_xywh, pred_s, pred_r, pred_l1234, pred_conf]

        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy0 = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy0.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(device)

        pred_l1234 = torch.exp(conv_raw_l1234) * out1_l
        pred_xmin = grid_xy[:, :, :, 0:1] * self.__stride + self.__stride / 2 - pred_l1234[:, :, :, 3:4]
        pred_ymin = grid_xy[:, :, :, 1:2] * self.__stride + self.__stride / 2 - pred_l1234[:, :, :, 0:1]
        pred_xmax = grid_xy[:, :, :, 0:1] * self.__stride + self.__stride / 2 + pred_l1234[:, :, :, 1:2]
        pred_ymax = grid_xy[:, :, :, 1:2] * self.__stride + self.__stride / 2 + pred_l1234[:, :, :, 2:3]

        pred_w = (pred_l1234[:, :, :, 1:2] + pred_l1234[:, :, :, 3:4])
        pred_h = (pred_l1234[:, :, :, 0:1] + pred_l1234[:, :, :, 2:3])
        pred_x = (pred_xmax + pred_xmin) / 2
        pred_y = (pred_ymax + pred_ymin) / 2
        pred_xywh = torch.cat([pred_x, pred_y, pred_w, pred_h], dim=-1)
        pred_s = (torch.clamp(torch.sigmoid(conv_raw_s), 0.01, 1) - 0.01) / (1 - 0.01)
        pred_s = (0.3 * out1_s + 0.7 * pred_s)
        pred_r = F.relu6(conv_raw_r + 3, inplace=True) / 6
        zero = torch.zeros_like(pred_r)
        one = torch.ones_like(pred_r)
        maskr = torch.where(pred_r > 0.9, zero, one)  # 0.8
        pred_s[:, :, :, 0:1] = pred_s[:, :, :, 0:1] * maskr
        pred_s[:, :, :, 1:2] = pred_s[:, :, :, 1:2] * maskr
        pred_s[:, :, :, 2:3] = pred_s[:, :, :, 2:3] * maskr
        pred_s[:, :, :, 3:4] = pred_s[:, :, :, 3:4] * maskr
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)

        pred_bbox = torch.cat([pred_xywh, pred_s, pred_r, pred_l1234, pred_conf, pred_prob], dim=-1)
        out_de = pred_bbox.view(-1, 4 + 5 + 4 + self.__nC + 1) if not self.training else pred_bbox
        return out_de