import sys
sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import utils_basic
import config.config as cfg

class HeatmapLoss(nn.Module):
    def __init__(self,  weight=None, alpha=2, beta=4, reduction='mean'):
        super(HeatmapLoss, self).__init__()
        self.alpha = alpha
        self.beta = beta
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        center_loss = -center_id * (1.0-inputs)**self.alpha * torch.log(inputs + 1e-14)
        other_loss = -other_id * (1 - targets)**self.beta * (inputs)**self.alpha * torch.log(1.0 - inputs + 1e-14)

        return center_loss + other_loss

class Smooth_Heatmap_Loss(nn.Module):
    def __init__(self,  weight=None, alpha=2, beta=4, reduction='mean'):
        super(Smooth_Heatmap_Loss, self).__init__()
        self.alpha = alpha
        self.beta = beta
        self.delta = 0.01
        self.num_class = cfg.DATA["NUM"]
    def forward(self, inputs, targets):
        inputs = torch.sigmoid(inputs)
        center_id = (targets == 1.0).float()
        other_id = (targets != 1.0).float()
        targets_smooth = targets * (1 - self.delta) + self.delta * 1.0 / self.num_class
        center_loss = -center_id * targets_smooth * (1.0 -inputs) ** self.alpha * torch.log(inputs + 1e-14) * 5
        other_loss = -other_id * (1 - targets_smooth) ** self.beta * (inputs) **self.alpha * torch.log(1.0 - inputs + 1e-14)
        return (center_loss + other_loss)

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)
        return loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.__strides = 4
        self.__scale_factor = cfg.SCALE_FACTOR

    def forward(self, p, p_d, label_bbox):
        loss, loss_iou, loss_cls, loss_a, loss_r, loss_txty, loss_twth = self.__cal_loss(p, p_d, label_bbox, self.__strides)
        return loss, loss_iou, loss_cls, loss_a, loss_r, loss_txty, loss_twth

    def smooth_l1_loss(self, input, target, beta=1. / 9):
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def __cal_loss(self, p, p_d, label, stride):
        batch_size, grid = p.shape[:2]
        img_size = stride * grid
        p_d_xywh = p_d[..., :4]
        p_d_a = p_d[..., 4:8]
        p_d_r = p_d[..., 8:9]
        p_cls = p[..., 9:]

        label_xywh = label[..., :4]
        label_txtytwth = label[..., 4:8]
        label_a = label[..., 8:12]
        label_r = label[..., 12:13]
        label_mask = label[..., 13:14]
        label_mix = label[..., 14:15]
        #label_angle = label[..., 15:16] ### If you need to change to the angle-based OBB representation method, calculate the loss of the angle.
        label_cls = label[..., 16:]

        xiou=[]
        # loss xiou
        if cfg.TRAIN["IOU_TYPE"] == 'GIOU':
            xiou = utils_basic.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        elif cfg.TRAIN["IOU_TYPE"] == 'CIOU':
            # print(p_d_xywh.shape, label_xywh.shape)
            xiou = utils_basic.CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        bbox_loss_scale = self.__scale_factor - (self.__scale_factor - 1.0) * label_xywh[..., 2:3] * label_xywh[...,3:4] / (img_size * img_size)
        loss_iou = bbox_loss_scale * (1.0 - xiou) * label_mix * label_mask #(1.0 - xiou) #xiou(-1,1),1-xiou(0,2)

        twth_loss_function = nn.SmoothL1Loss(reduction='none')
        txty_loss_function = nn.BCEWithLogitsLoss(reduction='none')  # nn.SmoothL1Loss(reduction='none')
        loss_txty = torch.sum(txty_loss_function(torch.sigmoid(p[..., 0:2]) * 1.05 - ((1.05 - 1) / 2), label_txtytwth[..., 0:2]), dim=-1, keepdim=True) * label_mask * label_mix
        loss_twth = torch.sum(twth_loss_function(p[..., 2:4], label_txtytwth[..., 2:4]), dim=-1, keepdim=True) * bbox_loss_scale * label_mix * label_mask

        loss_a = self.smooth_l1_loss(p_d_a, label_a) * label_mix * label_mask
        loss_r = self.smooth_l1_loss(p_d_r, label_r) * label_mix * label_mask

        # loss classes
        cls_loss_function = Smooth_Heatmap_Loss()
        loss_cls = cls_loss_function(p_cls, label_cls)

        loss_iou = (torch.sum(loss_iou)) / batch_size
        loss_cls = (torch.sum(loss_cls)) / batch_size
        loss_a = (torch.sum(loss_a)) / batch_size
        loss_r = 16 * (torch.sum(loss_r)) / batch_size
        loss_txty = (torch.sum(loss_txty)) / batch_size
        loss_twth = (torch.sum(loss_twth)) / batch_size
        loss = loss_iou + (loss_a + loss_r) + loss_cls + loss_txty + loss_twth

        return loss, loss_iou, loss_cls, loss_a, loss_r, loss_txty, loss_twth
