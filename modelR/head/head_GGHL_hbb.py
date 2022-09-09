import torch.nn as nn
import torch

class Head(nn.Module):
    def __init__(self, nC, stride):
        super(Head, self).__init__()
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, 4 + self.__nC + 1, nG, nG).permute(0, 2, 3, 1)
        p_de = self.__decode(p.clone())
        return (p, p_de)

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        conv_raw_l1234 = p[:, :, :, 0:4]
        conv_raw_conf = p[:, :, :, 4:5]
        conv_raw_prob = p[:, :, :, 5:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(device)

        pred_l1234 = torch.exp(conv_raw_l1234) * stride
        pred_xmin = grid_xy[:, :, :, 0:1] * stride + stride / 2 - pred_l1234[:, :, :, 3:4]
        pred_ymin = grid_xy[:, :, :, 1:2] * stride + (stride) / 2 - pred_l1234[:, :, :, 0:1]
        pred_xmax = grid_xy[:, :, :, 0:1] * stride + (stride) / 2 + pred_l1234[:, :, :, 1:2]
        pred_ymax = grid_xy[:, :, :, 1:2] * stride + (stride) / 2 + pred_l1234[:, :, :, 2:3]

        pred_w = (pred_l1234[:, :, :, 1:2] + pred_l1234[:, :, :, 3:4])
        pred_h = (pred_l1234[:, :, :, 0:1] + pred_l1234[:, :, :, 2:3])
        pred_x = (pred_xmax + pred_xmin)/2
        pred_y = (pred_ymax + pred_ymin)/2
        pred_xywh = torch.cat([pred_x, pred_y, pred_w, pred_h], dim=-1)
        pred_conf = torch.sigmoid(conv_raw_conf)
        pred_prob = torch.sigmoid(conv_raw_prob)


        pred_bbox = torch.cat([pred_xywh, pred_l1234, pred_conf, pred_prob], dim=-1)
        out = pred_bbox.view(-1, 4 + 4 + self.__nC + 1) if not self.training else pred_bbox
        return out
