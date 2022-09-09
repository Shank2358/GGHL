import sys

sys.path.append("../utils")
import torch
import torch.nn as nn
from utils import utils_basic
import config.config as cfg
import math

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)
    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)), self.__gamma)#gamma相当于Heatmap的alpha
        return loss

class Loss(nn.Module):
    def __init__(self):
        super(Loss, self).__init__()
        self.__strides = cfg.MODEL["STRIDES"]
        self.__scale_factor = cfg.SCALE_FACTOR
        self.__scale_factor_a = cfg.SCALE_FACTOR_A
        self.delta = 0.01
        self.num_class = cfg.DATA["NUM"]
        self.warmup = cfg.TRAIN["WARMUP_EPOCHS"]
        self.epoch = cfg.TRAIN["EPOCHS"]
        self.thresh_gh = 0.1

    def forward(self, p, p_d, label_sbbox, label_mbbox, label_lbbox, epoch, i):

        loss_s, loss_fg_s, loss_bg_s, loss_pos_s, loss_neg_s, loss_iou_s, loss_cls_s, loss_l_s = \
            self.__cal_loss(p[0], p_d[0], label_sbbox, int(self.__strides[0]), epoch, i)

        loss_m, loss_fg_m, loss_bg_m, loss_pos_m, loss_neg_m, loss_iou_m, loss_cls_m, loss_l_m = \
            self.__cal_loss(p[1], p_d[1], label_mbbox, int(self.__strides[1]), epoch, i)

        loss_l, loss_fg_l, loss_bg_l, loss_pos_l, loss_neg_l, loss_iou_l, loss_cls_l, loss_l_l = \
            self.__cal_loss(p[2], p_d[2], label_lbbox, int(self.__strides[2]), epoch, i)

        loss = loss_l + loss_m + loss_s
        loss_fg = loss_fg_s + loss_fg_m + loss_fg_l
        loss_bg = loss_bg_s + loss_bg_m + loss_bg_l
        loss_pos = loss_pos_s + loss_pos_m + loss_pos_l
        loss_neg = loss_neg_s + loss_neg_m + loss_neg_l

        loss_iou = loss_iou_s + loss_iou_m + loss_iou_l
        loss_cls = loss_cls_s + loss_cls_m + loss_cls_l
        loss_l = loss_l_s + loss_l_m + loss_l_l

        return loss, loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_l

    def smooth_l1_loss(self, input, target, beta=1. / 9, size_average=True):
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def __cal_loss(self, p, p_d, label, stride, epoch, iter):

        batch_size, grid = p.shape[:2]
        img_size = stride * grid

        label_xywh = label[..., :4]
        label_l1234 = label[..., 4:8]
        label_mask = label[..., 8:9]
        label_mix = label[..., 9:10]
        label_areaweight = label[..., 10:11]
        label_cls = label[..., 11:]

        p_d_xywh = p_d[..., :4]
        p_d_l = p_d[..., 4:8]
        p_conf = p[..., 4:5]
        p_cls = p[..., 5:]

        Focal = FocalLoss(gamma=2, alpha=1.0, reduction="none")
        BCE = nn.BCEWithLogitsLoss(reduction="none")

        gh_mask = torch.max(label_cls, dim=-1, keepdim=True)[0]
        label_noobj_mask = 1 - label_mask
        label_conf_smooth = (label_mask) * (1 - self.delta) + self.delta * 1.0 / 2
        label_cls_smooth = (label_cls != 0) * (1 - self.delta) + self.delta * 1.0 / self.num_class
        area_weight = label_areaweight + label_noobj_mask

        # iou
        xiou = []
        if cfg.TRAIN["IOU_TYPE"] == 'GIOU':
            xiou = utils_basic.GIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        elif cfg.TRAIN["IOU_TYPE"] == 'CIOU':
            xiou = utils_basic.CIOU_xywh_torch(p_d_xywh, label_xywh).unsqueeze(-1)
        bbox_loss_scale = self.__scale_factor - (self.__scale_factor - 1.0) \
                          * label_xywh[..., 2:3] * label_xywh[...,3:4] / (img_size * img_size)

        scores_iou = bbox_loss_scale * (1.0 - xiou) #xiou(-1,1),1-xiou(0,2)
        scores_loc = torch.exp(-1 * (scores_iou))
        scores_cls_loc = torch.sigmoid(p_cls) * scores_loc
        scores_cls_loc = -torch.log((1-scores_cls_loc)/(scores_cls_loc+1e-16)+1e-16)
        offset0 = scores_loc.detach()
        offset0 = torch.max(offset0, dim=-1, keepdim=True)[0]

        loss_fg = label_mask * Focal(input=p_conf, target=label_conf_smooth) * label_mix * ((gh_mask)+offset0)/2
        loss_bg = label_noobj_mask * Focal(input=p_conf, target=label_conf_smooth) * label_mix

        loss_pos = (label_cls != 0).float() * label_mask * BCE(input=scores_cls_loc, target=label_cls_smooth) * label_mix  * area_weight #-torch.log(scores_cls_loc + 1e-16)
        loss_neg = (1 - (label_cls != 0).float()) * label_mask * BCE(input=p_cls, target=label_cls_smooth) * label_mix * area_weight
        N = (torch.sum(label_mask.view(batch_size, -1), dim=-1) + 1e-16)
        N = torch.max(N, torch.ones(N.size(), device=N.device)).view(batch_size, 1, 1, 1)
        loss_fg = (torch.sum(loss_fg / N)) / batch_size * 2
        loss_bg = (torch.sum(loss_bg / N)) / batch_size * 2
        loss_pos = (torch.sum(loss_pos / N)) / batch_size
        loss_neg = (torch.sum(loss_neg / N)) / batch_size

        weight_cls = torch.sum((label_cls != 0).float() * torch.sigmoid(p_cls), dim=-1, keepdim=True)
        loss_iou = label_mask * scores_iou * label_mix * area_weight * (weight_cls + (gh_mask)) / 2
        SmoothL1 = nn.SmoothL1Loss(reduction='none')
        loss_l = label_mask * bbox_loss_scale * SmoothL1(p_d_l / stride, label_l1234 / stride) * label_mix * area_weight * (weight_cls + (gh_mask)) / 2
        loss_cls = label_mask * BCE(input=p_cls, target=label_cls_smooth) * label_mix * area_weight

        loss_iou = (torch.sum(loss_iou / N)) / batch_size
        loss_cls = (torch.sum(loss_cls / N)) / batch_size
        loss_l = 0.2 * (torch.sum(loss_l / N)) / batch_size

        loss = loss_fg + loss_bg + loss_pos + loss_neg + loss_iou + loss_l + loss_cls
        return loss, loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_l
