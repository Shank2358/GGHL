import torch.nn as nn
import torch
import torch.nn.functional as F
import config.config as cfg

class MSigmoid_approx(nn.Module):
    def __init__(self):
        super(MSigmoid_approx, self).__init__()
    def forward(self, input):
        return torch.sqrt(torch.sigmoid(input)*min(max(0,input+3),6)/6)

class Head(nn.Module):
    def __init__(self, nC, anchors, stride, fact):
        super(Head, self).__init__()
        self.fact = fact
        self.__anchors = anchors
        self.__nA = len(anchors)
        self.__nC = nC
        self.__stride = stride
        self.__MSigmoid_approx = MSigmoid_approx()

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, self.__nA, 5 + 5 + self.__nC, nG, nG).permute(0, 3, 4, 1, 2)##############xywhc+a1-a4+r+class
        p_de = self.__decode(p.clone())
        return (p, p_de)

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        anchors = (1.0 * self.__anchors).to(device)#################608
        conv_raw_dxdy = p[:, :, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, :, 2:4]
        conv_raw_a = p[:, :, :, :, 4:8]
        conv_raw_r = p[:, :, :, :, 8:9]
        conv_raw_conf = p[:, :, :, :, 9:10]
        conv_raw_prob = p[:, :, :, :, 10:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).unsqueeze(3).repeat(batch_size, 1, 1, cfg.MODEL["ANCHORS_PER_SCLAE"], 1).float().to(device)
        # pred_xy = (torch.sigmoid(conv_raw_dxdy) + grid_xy) * stride
        pred_xy = (torch.sigmoid(conv_raw_dxdy) * 1.05 - ((1.05 - 1) / 2) + grid_xy) * stride
        pred_wh = (torch.exp(conv_raw_dwdh) * anchors ) * stride #* wh_new
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        # pred_a = torch.sigmoid(conv_raw_a)
        # pred_r = torch.sigmoid(conv_raw_r)
        pred_a = F.relu6(conv_raw_a + 3, inplace=True) / 6
        pred_r = F.relu6(conv_raw_r + 3, inplace=True) / 6

        maskr = pred_r
        zero = torch.zeros_like(maskr)
        one = torch.ones_like(maskr)
        maskr = torch.where(maskr > 0.85, zero, one)
        pred_a[:, :, :, :, 0:1] = pred_a[:, :, :, :, 0:1] * maskr
        pred_a[:, :, :, :, 1:2] = pred_a[:, :, :, :, 1:2] * maskr
        pred_a[:, :, :, :, 2:3] = pred_a[:, :, :, :, 2:3] * maskr
        pred_a[:, :, :, :, 3:4] = pred_a[:, :, :, :, 3:4] * maskr

        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)
        pred_bbox = torch.cat([pred_xywh, pred_a, pred_r, pred_conf, pred_prob], dim=-1)
        return pred_bbox.view(-1, 5 + 5 + self.__nC) if not self.training else pred_bbox