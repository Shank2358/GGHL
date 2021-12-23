import torch.nn as nn
import torch

class Head(nn.Module):
    def __init__(self, nC, stride):
        super(Head, self).__init__()
        self.__nC = nC
        self.__stride = stride

    def forward(self, p):
        bs, nG = p.shape[0], p.shape[-1]
        p = p.view(bs, 4 + 5 + self.__nC, nG, nG).permute(0, 2, 3, 1)
        p_de = self.__decode(p.clone())
        return (p, p_de)

    def __decode(self, p):
        batch_size, output_size = p.shape[:2]
        device = p.device
        stride = self.__stride
        conv_raw_dxdy = p[:, :, :, 0:2]
        conv_raw_dwdh = p[:, :, :, 2:4]
        conv_raw_a = p[:, :, :, 4:8]
        conv_raw_r = p[:, :, :, 8:9]
        conv_raw_prob = p[:, :, :, 9:]
        y = torch.arange(0, output_size).unsqueeze(1).repeat(1, output_size)
        x = torch.arange(0, output_size).unsqueeze(0).repeat(output_size, 1)
        grid_xy = torch.stack([x, y], dim=-1)
        grid_xy = grid_xy.unsqueeze(0).repeat(batch_size, 1, 1, 1).float().to(device)
        pred_xy = (torch.sigmoid(conv_raw_dxdy) * 1.05 - ((1.05 - 1) / 2) + grid_xy) * stride
        pred_wh = torch.exp(conv_raw_dwdh) * stride
        pred_xywh = torch.cat([pred_xy, pred_wh], dim=-1)
        pred_a = torch.sigmoid(conv_raw_a)
        pred_r = torch.sigmoid(conv_raw_r)

        maskr = pred_r
        zero = torch.zeros_like(maskr)
        one = torch.ones_like(maskr)
        maskr = torch.where(maskr > 0.9, zero, one)
        pred_a[:, :, :, 0:1] = pred_a[:, :, :, 0:1] * maskr
        pred_a[:, :, :, 1:2] = pred_a[:, :, :, 1:2] * maskr
        pred_a[:, :, :, 2:3] = pred_a[:, :, :, 2:3] * maskr
        pred_a[:, :, :, 3:4] = pred_a[:, :, :, 3:4] * maskr

        pred_prob = torch.sigmoid(conv_raw_prob)

        import matplotlib
        matplotlib.use('TkAgg')
        import matplotlib.pyplot as plt
        plt.figure("Image4")  # 图像窗口名称
        plt.imshow(torch.max(pred_prob, dim=-1)[0].squeeze(0).detach().cpu())
        plt.show()

        pred_bbox = torch.cat([pred_xywh, pred_a, pred_r, pred_prob], dim=-1)
        return pred_bbox.view(-1, 4 + 5 + self.__nC) if not self.training else pred_bbox
