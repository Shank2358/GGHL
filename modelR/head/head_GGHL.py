import torch.nn as nn
import torch
import torch.nn.functional as F

class Head(nn.Module):
    def __init__(self, nC, stride):
        super(Head, self).__init__()
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, 4 + 5 + self.__nC + 1, nG, nG).permute(0, 2, 3, 1)##############xywhc+a1-a4+r+class
        p_de = self.__decode(p.clone())
        return p, p_de

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        conv_raw_l = p[:, :, :, 0:4]
        conv_raw_s = p[:, :, :, 4:8]
        conv_raw_r = p[:, :, :, 8:9]
        conv_raw_conf = p[:, :, :, 9:10]
        conv_raw_prob = p[:, :, :, 10:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(device)

        pred_l = conv_raw_l ** 2 * stride
        pred_xmin = grid_xy[:, :, :, 0:1] * stride + (stride) / 2 - pred_l[:, :, :, 3:4]
        pred_ymin = grid_xy[:, :, :, 1:2] * stride + (stride) / 2 - pred_l[:, :, :, 0:1]
        pred_xmax = grid_xy[:, :, :, 0:1] * stride + (stride) / 2 + pred_l[:, :, :, 1:2]
        pred_ymax = grid_xy[:, :, :, 1:2] * stride + (stride) / 2 + pred_l[:, :, :, 2:3]

        pred_w = (pred_l[:, :, :, 1:2] + pred_l[:, :, :, 3:4])
        pred_h = (pred_l[:, :, :, 0:1] + pred_l[:, :, :, 2:3])
        pred_x = (pred_xmax + pred_xmin)/2
        pred_y = (pred_ymax + pred_ymin)/2
        pred_xywh = torch.cat([pred_x, pred_y, pred_w, pred_h], dim=-1)
        pred_s = (torch.clamp(torch.sigmoid(conv_raw_s), 0.01,1)-0.01)/(1-0.01)
        pred_r = torch.sigmoid(conv_raw_r)#F.relu6(conv_raw_r + 3, inplace=True) / 6
        #self.grid_sens = 0.05
        #pred_s = self.grid_sens * torch.sigmoid(conv_raw_s) - 0.5 * (self.grid_sens - 1.0)
        #pred_r = self.grid_sens * torch.sigmoid(conv_raw_r) - 0.5 * (self.grid_sens - 1.0)
        maskr = pred_r
        zero = torch.zeros_like(maskr)
        one = torch.ones_like(maskr)
        maskr = torch.where(maskr > 0.9, zero, one) #0.8
        pred_s[:, :, :, 0:1] = pred_s[:, :, :, 0:1] * maskr
        pred_s[:, :, :, 1:2] = pred_s[:, :, :, 1:2] * maskr
        pred_s[:, :, :, 2:3] = pred_s[:, :, :, 2:3] * maskr
        pred_s[:, :, :, 3:4] = pred_s[:, :, :, 3:4] * maskr
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)

        pred_bbox = torch.cat([pred_xywh, pred_s, pred_r, pred_l, pred_conf, pred_prob], dim=-1)
        out = pred_bbox.view(-1, 4 + 5 + 4 + self.__nC + 1) if not self.training else pred_bbox
        return out
