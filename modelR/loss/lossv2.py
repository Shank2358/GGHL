import torch
import torch.nn as nn
from utils import utils_basic
import config.config as cfg

class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=1.0, reduction="mean"):
        super(FocalLoss, self).__init__()
        self.__gamma = gamma
        self.__alpha = alpha
        self.__loss = nn.BCEWithLogitsLoss(reduction=reduction)

    def forward(self, input, target):
        loss = self.__loss(input=input, target=target)
        loss *= self.__alpha * torch.pow(torch.abs(target - torch.sigmoid(input)),
                                         self.__gamma)  # gamma相当于Heatmap的alpha
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

    def forward(self, p1, p1_d, p2, p2_d, label_sbbox, label_mbbox, label_lbbox, epoch, i):

        loss_s, loss_fg_s, loss_bg_s, loss_pos_s, loss_neg_s, loss_iou_s, loss_cls_s, loss_s_s, loss_r_s, loss_l_s = \
            self.__cal_loss(p1[0], p1_d[0], p2[0], p2_d[0], label_sbbox, int(self.__strides[0]), epoch, i)

        loss_m, loss_fg_m, loss_bg_m, loss_pos_m, loss_neg_m, loss_iou_m, loss_cls_m, loss_s_m, loss_r_m, loss_l_m = \
            self.__cal_loss(p1[1], p1_d[1], p2[1], p2_d[1], label_mbbox, int(self.__strides[1]), epoch, i)

        loss_l, loss_fg_l, loss_bg_l, loss_pos_l, loss_neg_l, loss_iou_l, loss_cls_l, loss_s_l, loss_r_l, loss_l_l = \
            self.__cal_loss(p1[2], p1_d[2], p2[2], p2_d[2], label_lbbox, int(self.__strides[2]), epoch, i)

        loss = loss_l + loss_m + loss_s
        loss_fg = loss_fg_s + loss_fg_m + loss_fg_l
        loss_bg = loss_bg_s + loss_bg_m + loss_bg_l
        loss_pos = loss_pos_s + loss_pos_m + loss_pos_l
        loss_neg = loss_neg_s + loss_neg_m + loss_neg_l

        loss_iou = loss_iou_s + loss_iou_m + loss_iou_l
        loss_cls = loss_cls_s + loss_cls_m + loss_cls_l
        loss_s = loss_s_s + loss_s_m + loss_s_l
        loss_r = loss_r_s + loss_r_m + loss_r_l
        loss_l = loss_l_s + loss_l_m + loss_l_l

        return loss, loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_s, loss_r, loss_l

    def smooth_l1_loss(self, input, target, beta=1. / 9, size_average=True):
        n = torch.abs(input - target)
        cond = n < beta
        loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
        return loss

    def __cal_loss(self, p1, p1_d, p2, p2_d, label, stride, epoch, iter):

        batch_size, grid = p1.shape[:2]
        img_size = stride * grid

        label_xywh = label[..., :4]
        label_l1234 = label[..., 4:8]
        label_a = label[..., 8:12]
        label_r = label[..., 12:13]
        label_mix = label[..., 14:15]
        label_cls = label[..., 16:]

        MSE = nn.MSELoss(reduction='none')
        Focal = FocalLoss(gamma=2, alpha=1.0, reduction="none")
        BCE = nn.BCEWithLogitsLoss(reduction="none")
        SmoothL1 = nn.SmoothL1Loss(reduction='none')
        # KL = nn.KLDivLoss(reduction='sum')

        obj_mask = (label[..., 13:14] == 1).float()  ###二值 positive_mask
        noobj_mask = 1 - (label[..., 13:14] != 0).float()  ###二值  negative_mask
        fuzzy_mask = 1 - obj_mask - noobj_mask
        gh = torch.max(label_cls, dim=-1, keepdim=True)[0]  # positive_gaussian
        #gh_obj = obj_mask * gh
        gh_fuzzy = fuzzy_mask * gh

        self.delta_conf = 0.01
        label_conf_smooth = obj_mask * (1 - self.delta_conf) + self.delta_conf * 1.0 / 2  ###############
        label_cls_smooth = (label_cls != 0).float() * (1 - self.delta) + self.delta * 1.0 / self.num_class

        area_weight = label[..., 15:16] + (label[..., 15:16] == 0).float()

        bbox_loss_scale = self.__scale_factor - (self.__scale_factor - 1.0) \
                          * label_xywh[..., 2:3] * label_xywh[..., 3:4] / (img_size * img_size)
        N = (torch.sum(obj_mask.view(batch_size, -1), dim=-1) + 1e-16)
        N = torch.max(N, torch.ones(N.size(), device=N.device)).view(batch_size, 1, 1, 1)

        p1_d_xywh = p1_d[..., :4]
        p1_d_s = p1_d[..., 4:8]
        p1_d_r = p1_d[..., 8:9]
        p1_d_l = p1_d[..., 9:13]
        p1_conf = p1[..., 9:10]

        # iou
        xiou1 = []
        if cfg.TRAIN["IOU_TYPE"] == 'GIOU':
            xiou1 = utils_basic.GIOU_xywh_torch(p1_d_xywh, label_xywh).unsqueeze(-1)
        elif cfg.TRAIN["IOU_TYPE"] == 'CIOU':
            xiou1 = utils_basic.CIOU_xywh_torch(p1_d_xywh, label_xywh).unsqueeze(-1)
        scores_iou1 = bbox_loss_scale * (1.0 - torch.clamp(xiou1, 0, 1))
        scores_obb1 = torch.sum(MSE(p1_d_s, label_a), dim=-1, keepdim=True)
        scores_area1 = MSE(p1_d_r, label_r)
        scores_loc1 = torch.exp(-1 * (scores_iou1 + scores_obb1 + scores_area1))
        offset01 = scores_loc1.detach()
        offset01 = torch.max(offset01, dim=-1, keepdim=True)[0]

        bg_mask1 = noobj_mask + fuzzy_mask * ((gh + offset01) / 2 < 0.3).float() * (1 - fuzzy_mask * (gh + offset01) / 2)
        fg_mask1 = obj_mask * ((gh + offset01) / 2 >= 0.3).float()
        loss_fg1 = fg_mask1 * Focal(input=p1_conf, target=label_conf_smooth) * label_mix * (gh + offset01) / 2
        loss_bg1 = bg_mask1 * Focal(input=p1_conf, target=label_conf_smooth) * label_mix

        # loss_fg1 = obj_mask * Focal(input=p1_conf, target=label_conf_smooth) * label_mix * (gh_obj + offset01) / 2
        # loss_bg1 = noobj_mask * Focal(input=p1_conf, target=label_conf_smooth) * label_mix

        loss_iou1 = fg_mask1 * scores_iou1 * label_mix * area_weight
        loss_s1 = fg_mask1 * scores_obb1 * label_mix * area_weight
        loss_r1 = fg_mask1 * scores_area1 * label_mix * area_weight
        loss_l1 = fg_mask1 * bbox_loss_scale * SmoothL1(p1_d_l / stride, label_l1234 / stride) * label_mix * area_weight

        loss_fg1 = (torch.sum(loss_fg1 / N)) / batch_size * 2
        loss_bg1 = (torch.sum(loss_bg1 / N)) / batch_size * 2
        loss_iou1 = (torch.sum(loss_iou1 / N)) / batch_size
        loss_s1 = (torch.sum(loss_s1 / N)) / batch_size
        loss_r1 = 16 * (torch.sum(loss_r1 / N)) / batch_size
        loss_l1 = 0.2 * (torch.sum(loss_l1 / N)) / batch_size
        ######################

        p2_d_xywh = p2_d[..., :4]
        p2_d_s = p2_d[..., 4:8]
        p2_d_r = p2_d[..., 8:9]
        p2_d_l = p2_d[..., 9:13]
        p2_conf = p2[..., 9:10]
        p2_cls = p2[..., 10:]

        xiou2 = []
        if cfg.TRAIN["IOU_TYPE"] == 'GIOU':
            xiou2 = utils_basic.GIOU_xywh_torch(p2_d_xywh, label_xywh).unsqueeze(-1)
        elif cfg.TRAIN["IOU_TYPE"] == 'CIOU':
            xiou2 = utils_basic.CIOU_xywh_torch(p2_d_xywh, label_xywh).unsqueeze(-1)
        scores_iou2 = bbox_loss_scale * (1.0 - torch.clamp(xiou2, 0, 1))
        scores_obb2 = torch.sum(MSE(p2_d_s, label_a), dim=-1, keepdim=True)
        scores_area2 = MSE(p2_d_r, label_r)
        scores_loc2 = torch.exp(-1 * (scores_iou2 + scores_obb2 + scores_area2))
        #scores_cls_loc2 = torch.sigmoid(p2_cls) * scores_loc2
        #scores_cls_loc2 = -torch.log((1 - scores_cls_loc2) / (scores_cls_loc2 + 1e-16) + 1e-16)

        offset02 = scores_loc2.detach()
        #offset02 = torch.max(offset02, dim=-1, keepdim=True)[0]

        bg_mask2 = noobj_mask + fuzzy_mask * ((gh + offset02) / 2 < 0.3).float() * (1 - fuzzy_mask * (gh + offset02) / 2)
        fg_mask2 = obj_mask * ((gh + offset02) / 2 >= 0.3).float()
        loss_fg2 = fg_mask2 * Focal(input=p2_conf, target=label_conf_smooth) * label_mix * (gh + offset02) / 2
        loss_bg2 = bg_mask2 * Focal(input=p2_conf, target=label_conf_smooth) * label_mix

        # loss_fg2 = obj_mask * Focal(input=p2_conf, target=label_conf_smooth) * label_mix * (gh_obj + offset02) / 2
        # loss_bg2 = noobj_mask * Focal(input=p2_conf, target=label_conf_smooth) * label_mix
        self.zeta = 0.3
        w_neg2 = (1 - (label_cls != 0).float()) * torch.sigmoid(p2_cls).detach() 
        mask_w_neg2 = (w_neg2 > self.zeta).float()
        w_neg2 = (1 - mask_w_neg2) * w_neg2 + mask_w_neg2
        
        loss_pos2 = (label_cls != 0).float() * obj_mask * BCE(input=p2_cls, target = label_cls_smooth * offset02) * label_mix * area_weight
        loss_neg2 = (1 - (label_cls != 0).float()) * obj_mask * BCE(input=p2_cls, target=label_cls_smooth) * label_mix * area_weight
        weight_cls2 = torch.sum((label_cls != 0).float() * torch.sigmoid(p2_cls.detach()), dim=-1, keepdim=True)
        loss_iou2 = fg_mask2 * scores_iou2 * label_mix * area_weight * (weight_cls2 + gh) / 2
        loss_s2 = fg_mask2 * scores_obb2 * label_mix * area_weight * (weight_cls2 + gh) / 2
        loss_r2 = fg_mask2 * scores_area2 * label_mix * area_weight * (weight_cls2 + gh) / 2
        loss_l2 = fg_mask2 * bbox_loss_scale * SmoothL1(p2_d_l / stride, label_l1234 / stride) * label_mix * area_weight * (weight_cls2 + gh) / 2
        loss_cls2 = fg_mask2 * BCE(input=p2_cls, target=label_cls_smooth) * label_mix * area_weight

        loss_fg2 = (torch.sum(loss_fg2 / N)) / batch_size * 2
        loss_bg2 = (torch.sum(loss_bg2 / N)) / batch_size * 2
        loss_pos2 = (torch.sum(loss_pos2 / N)) / batch_size * (self.num_class-1)
        loss_neg2 = (torch.sum(loss_neg2 / N)) / batch_size
        loss_iou2 = (torch.sum(loss_iou2 / N)) / batch_size
        loss_cls2 = (torch.sum(loss_cls2 / N)) / batch_size
        loss_s2 = (torch.sum(loss_s2 / N)) / batch_size
        loss_r2 = 16 * (torch.sum(loss_r2 / N)) / batch_size
        loss_l2 = 0.2 * (torch.sum(loss_l2 / N)) / batch_size
        #############################################

        loss_fg = (loss_fg1 + loss_fg2) / 2
        loss_bg = (loss_bg1 + loss_bg2) / 2
        loss_pos = loss_pos2 * 2
        loss_neg = loss_neg2 * 2
        loss_cls = loss_cls2
        loss_iou = (loss_iou1 + loss_iou2) / 2
        loss_s = (loss_s1 + loss_s2) / 2
        loss_r = (loss_r1 + loss_r2) / 2
        loss_l = (loss_l1 + loss_l2) / 2

        loss = loss_fg + loss_bg + loss_iou + (loss_s + loss_r) + loss_pos + loss_neg + loss_l #+ loss_cls
        return loss, loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_s, loss_r, loss_l