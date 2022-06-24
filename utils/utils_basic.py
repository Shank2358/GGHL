#coding=utf-8
import os
import sys
sys.path.append("..")
import math
import numpy as np
import random
import torch
import config.config as cfg
from shapely.geometry import Polygon, MultiPoint  # 多边形

def init_seeds(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)

def xyxy2xywh(x):
    # Convert bounding box format from [x1, y1, x2, y2] to [x, y, w, h]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)

    y[:, 0] = (x[:, 0] + x[:, 2]) / 2.0
    y[:, 1] = (x[:, 1] + x[:, 3]) / 2.0
    y[:, 2] = x[:, 2] - x[:, 0]
    y[:, 3] = x[:, 3] - x[:, 1]
    return y

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y

def iou_xywh_numpy(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)
    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    boxes1 = np.concatenate([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                             boxes1[..., :2] + boxes1[..., 2:] * 0.5], axis=-1)
    boxes2 = np.concatenate([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                             boxes2[..., :2] + boxes2[..., 2:] * 0.5], axis=-1)

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return 1.0 * inter_area / union_area

def iou_xyxy_numpy(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    return 1.0 * inter_area / union_area

def diou_xyxy_numpy(boxes1, boxes2):
    boxes1 = np.array(boxes1)
    boxes2 = np.array(boxes2)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1和boxes2相交部分的左上角坐标、右下角坐标
    left_up = np.maximum(boxes1[..., :2], boxes2[..., :2])
    right_down = np.minimum(boxes1[..., 2:], boxes2[..., 2:])

    # 计算出boxes1和boxes2相交部分的宽、高
    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = np.maximum(right_down - left_up, 0.0)
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = np.minimum(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = np.maximum(boxes1[..., 2:], boxes2[..., 2:])
    enclose_section = np.maximum(enclose_right_down - enclose_left_up, np.zeros_like(enclose_right_down))
    enclose_c2 = np.power(enclose_section[..., 0], 2) + np.power(enclose_section[..., 1], 2)

    boxes1 = np.concatenate((0.5 * (boxes1[..., 0:1] + boxes1[..., 2:3]), 0.5 * (boxes1[..., 1:2] + boxes1[..., 3:]),
                             (boxes1[..., 2:3] - boxes1[..., 0:1]), (boxes1[..., 3:] - boxes1[..., 1:2])), axis=-1)
    boxes2 = np.concatenate((0.5 * (boxes2[..., 0:1] + boxes2[..., 2:3]), 0.5 * (boxes2[..., 1:2] + boxes2[..., 3:]),
                             (boxes2[..., 2:3] - boxes2[..., 0:1]), (boxes2[..., 3:] - boxes2[..., 1:2])), axis=-1)

    p2 = np.power(boxes1[..., 0] - boxes2[..., 0], 2) + np.power(boxes1[..., 1] - boxes2[..., 1], 2)

    DIOU = IOU - 1.0 * p2 / enclose_c2
    return DIOU

def polygen_iou_xy4_numpy(boxes1, boxes2):
    #print(boxes2.shape)
    num = boxes2.shape[0]
    if num == 0:
        iou_out = []
    else:
        boxes1 = boxes1.reshape(-1, 4, 2)
        boxes2 = boxes2.reshape(-1, 4, 2)
        #print(boxes1.shape, boxes2.shape)
        iou = np.zeros(num)
        for i in range(0, num):
            #print("num",num,i)
            poly1 = Polygon(boxes1[0,:,:]).convex_hull
            #print("uuuuuuu,",boxes2.shape)
            poly2 = Polygon(boxes2[i,:,:]).convex_hull
            union_poly = np.concatenate((boxes1[0,:,:], boxes2[i,:,:]),axis=0)
            if poly1.intersects(poly2):  # 如果两四边形相交
                inter_area = poly1.intersection(poly2).area  # 相交面积
                union_area = MultiPoint(union_poly).convex_hull.area
                iou[i] = float(inter_area) / union_area
        iou_out = iou
    return np.array(iou_out)

def polygen_iou_xy4_numpy_eval(boxes1, boxes2):
    boxes1 = boxes1.reshape(4, 2)
    boxes2 = boxes2.reshape(4, 2)
    poly1 = Polygon(boxes1).convex_hull
    poly2 = Polygon(boxes2).convex_hull
    union_poly = np.concatenate((boxes1, boxes2),axis=0)
    iou=0
    if poly1.intersects(poly2):  # 如果两四边形相交
        inter_area = poly1.intersection(poly2).area  # 相交面积
        union_area = MultiPoint(union_poly).convex_hull.area
        iou = float(inter_area) / union_area
    return iou

def polygen_iou_xy4_numpy1(boxes1, boxes2):############loss
    size1 = boxes1.shape
    num = size1[0]*size1[1]*size1[2]*size1[3]
    boxes1 = boxes1.cpu().detach().numpy()
    boxes2 = boxes2.cpu().detach().numpy()
    boxes1 = boxes1.reshape(-1, 4, 2)
    boxes2 = boxes2.reshape(-1, 4, 2)
    iou = np.zeros(num)
    for i in range(0, num):
        poly1 = Polygon(boxes1[i,:,:]).convex_hull
        poly2 = Polygon(boxes2[i,:,:]).convex_hull
        union_poly = np.concatenate((boxes1[i,:,:], boxes2[i,:,:]))
        if poly1.intersects(poly2):  # 如果两四边形相交
            inter_area = poly1.intersection(poly2).area  # 相交面积
            union_area = MultiPoint(union_poly).convex_hull.area
            iou[i] = float(inter_area) / union_area
    iou_out = iou.reshape((size1[0],size1[1],size1[2],size1[3]))
    return torch.tensor(iou_out)

def polygen_iou_xy4_torch(boxes1, boxes2):
    size1 = boxes1.shape
    num = size1[0]*size1[1]*size1[2]*size1[3]
    #boxes1 = boxes1.cpu().detach().numpy()
    #boxes2 = boxes2.cpu().detach().numpy()
    boxes1 = boxes1.view(-1, 4, 2)
    print(boxes1.shape)
    boxes2 = boxes2.view(-1, 4, 2)
    iou = torch.zeros(num)
    for i in range(0, num):
        poly1 = Polygon(boxes1[i,:,:]).convex_hull
        poly2 = Polygon(boxes2[i,:,:]).convex_hull
        union_poly = torch.cat((boxes1[i,:,:], boxes2[i,:,:]))
        if poly1.intersects(poly2):  # 如果两四边形不相交
            inter_area = poly1.intersection(poly2).area  # 相交面积
            union_area = MultiPoint(union_poly).convex_hull.area
            iou[i] = float(inter_area) / union_area
    iou_out = iou.view(size1[0],size1[1],size1[2],size1[3])
    return iou_out

def polygen_iou_xy4_torch1(boxes1, boxes2):
    import cv2

    size1 = boxes1.shape
    num = size1[0]*size1[1]*size1[2]*size1[3]
    #boxes1 = boxes1.cpu().detach().numpy()
    #boxes2 = boxes2.cpu().detach().numpy()
    boxes1 = boxes1.view(-1, 4, 2).unsqueeze(1)
    boxes2 = boxes2.view(-1, 4, 2).unsqueeze(1)

    im = torch.zeros(num, 10, 10)
    im1 = torch.zeros(num, 10, 10)

    original_grasp_mask = cv2.fillPoly(im, boxes2, 255)
    print(original_grasp_mask.shape)
    prediction_grasp_mask = cv2.fillPoly(im1, boxes2, 255)
    masked_and = cv2.bitwise_and(original_grasp_mask, prediction_grasp_mask, mask=im)
    masked_or = cv2.bitwise_or(original_grasp_mask, prediction_grasp_mask)

    or_area = torch.sum(torch.float32(torch.gt(masked_or, 0)))
    and_area = torch.sum(torch.float32(torch.gt(masked_and, 0)))
    IOU = and_area / or_area

    iou = torch.zeros(num)
    for i in range(0, num):
        poly1 = Polygon(boxes1[i,:,:]).convex_hull
        poly2 = Polygon(boxes2[i,:,:]).convex_hull
        union_poly = torch.cat((boxes1[i,:,:], boxes2[i,:,:]))
        if poly1.intersects(poly2):  # 如果两四边形不相交
            inter_area = poly1.intersection(poly2).area  # 相交面积
            union_area = MultiPoint(union_poly).convex_hull.area
            iou[i] = float(inter_area) / union_area
    iou_out = iou.view(size1[0],size1[1],size1[2],size1[3])
    return iou_out


def iou_xyxy_torch(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(xmin, ymin, xmax, ymax)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """
    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

def iou_xywh_torch(boxes1, boxes2):
    """
    :param boxes1: boxes1和boxes2的shape可以不相同，但是需要满足广播机制，且需要是Tensor
    :param boxes2: 且需要保证最后一维为坐标维，以及坐标的存储结构为(x, y, w, h)
    :return: 返回boxes1和boxes2的IOU，IOU的shape为boxes1和boxes2广播后的shape[:-1]
    """

    boxes1_area = boxes1[..., 2] * boxes1[..., 3]
    boxes2_area = boxes2[..., 2] * boxes2[..., 3]

    # 分别计算出boxes1和boxes2的左上角坐标、右下角坐标
    # 存储结构为(xmin, ymin, xmax, ymax)，其中(xmin,ymin)是bbox的左上角坐标，(xmax,ymax)是bbox的右下角坐标
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    # 计算出boxes1与boxes1相交部分的左上角坐标、右下角坐标
    left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])

    # 因为两个boxes没有交集时，(right_down - left_up) < 0，所以maximum可以保证当两个boxes没有交集时，它们之间的iou为0
    inter_section = torch.max(right_down - left_up, torch.zeros_like(right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area
    return IOU

def GIOU_xywh_torch(boxes1, boxes2):
    """
     https://arxiv.org/abs/1902.09630
    boxes1(boxes2)' shape is [..., (x,y,w,h)].The size is for original image.
    """
    # xywh->xyxy
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / (union_area +1e-16)

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose_section = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
    enclose_area = enclose_section[..., 0] * enclose_section[..., 1]

    GIOU = IOU - 1.0 * (enclose_area - union_area) / (enclose_area+1e-16)
    return GIOU

#DIOU和CIOU
def CIOU_xywh_torch1(boxes1, boxes2):
    # xywh->xyxy
    p2 = torch.pow(boxes1[..., 0] - boxes2[..., 0], 2) + torch.pow(boxes1[..., 1] - boxes2[..., 1], 2)

    # 增加av。分母boxes2[..., 3]可能为0，所以加上除0保护防止nan。
    atan1 = torch.atan(boxes1[..., 2] / boxes1[..., 3])
    temp_a = torch.where(boxes2[..., 3] > 0.0, boxes2[..., 3], boxes2[..., 3] + 1.0)
    atan2 = torch.atan(boxes2[..., 2] / temp_a)
    v = 4.0 * torch.pow(atan1 - atan2, 2) / (math.pi ** 2)


    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])  # 内框的左上
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    IOU = 1.0 * inter_area / union_area

    enclose_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    enclose_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    enclose_section = torch.max(enclose_right_down - enclose_left_up, torch.zeros_like(enclose_right_down))
    enclose_c2 = torch.pow(enclose_section[..., 0], 2) + torch.pow(enclose_section[..., 1], 2)

    alpha = v / (1 - IOU + v)
    CIOU = IOU - 1.0 * p2 / enclose_c2 - 1.0 * alpha * v

    return CIOU

def CIOU_xywh_torch(boxes1,boxes2):
    
    #cal CIOU of two boxes or batch boxes
    #:param boxes1:[xmin,ymin,xmax,ymax] or[[xmin,ymin,xmax,ymax],[xmin,ymin,xmax,ymax],...]
    #:param boxes2:[xmin,ymin,xmax,ymax]
    #:return:
    
    # xywh->xyxy
    boxes1 = torch.cat([boxes1[..., :2] - boxes1[..., 2:] * 0.5,
                        boxes1[..., :2] + boxes1[..., 2:] * 0.5], dim=-1)
    boxes2 = torch.cat([boxes2[..., :2] - boxes2[..., 2:] * 0.5,
                        boxes2[..., :2] + boxes2[..., 2:] * 0.5], dim=-1)

    boxes1 = torch.cat([torch.min(boxes1[..., :2], boxes1[..., 2:]),
                        torch.max(boxes1[..., :2], boxes1[..., 2:])], dim=-1)
    boxes2 = torch.cat([torch.min(boxes2[..., :2], boxes2[..., 2:]),
                        torch.max(boxes2[..., :2], boxes2[..., 2:])], dim=-1)

    boxes1_area = (boxes1[..., 2] - boxes1[..., 0]) * (boxes1[..., 3] - boxes1[..., 1])
    boxes2_area = (boxes2[..., 2] - boxes2[..., 0]) * (boxes2[..., 3] - boxes2[..., 1])

    inter_left_up = torch.max(boxes1[..., :2], boxes2[..., :2])
    inter_right_down = torch.min(boxes1[..., 2:], boxes2[..., 2:])
    inter_section = torch.max(inter_right_down - inter_left_up, torch.zeros_like(inter_right_down))
    inter_area = inter_section[..., 0] * inter_section[..., 1]
    union_area = boxes1_area + boxes2_area - inter_area
    ious = 1.0 * inter_area / (union_area+1e-16)

    # cal outer boxes
    outer_left_up = torch.min(boxes1[..., :2], boxes2[..., :2])
    outer_right_down = torch.max(boxes1[..., 2:], boxes2[..., 2:])
    outer = torch.max(outer_right_down - outer_left_up, torch.zeros_like(inter_right_down))
    # outer_diagonal_line = torch.pow(outer[...,0]+outer[...,1])
    outer_diagonal_line = torch.pow(outer[..., 0], 2) + torch.pow(outer[..., 1], 2)
    # outer_diagonal_line = torch.sum(torch.pow(outer, 2), axis=-1)

    # cal center distance
    boxes1_center = (boxes1[..., :2] +  boxes1[...,2:]) * 0.5
    boxes2_center = (boxes2[..., :2] +  boxes2[...,2:]) * 0.5
    center_dis = torch.pow(boxes1_center[...,0]-boxes2_center[...,0], 2) +\
                 torch.pow(boxes1_center[...,1]-boxes2_center[...,1], 2)

    # cal penalty term
    # cal width,height
    boxes1_size = torch.max(boxes1[..., 2:] - boxes1[..., :2], torch.zeros_like(inter_right_down))
    boxes2_size = torch.max(boxes2[..., 2:] - boxes2[..., :2], torch.zeros_like(inter_right_down))
    v = (4 / (math.pi ** 2)) * torch.pow(
            torch.atan((boxes1_size[...,0]/torch.clamp(boxes1_size[...,1],min = 1e-6))) -
            torch.atan((boxes2_size[..., 0] / torch.clamp(boxes2_size[..., 1],min = 1e-6))), 2)
    alpha = v / (1-ious+v + 1e-16)

    #cal ciou
    cious = ious - (center_dis / (outer_diagonal_line+1e-16) + alpha*v)
    return cious


def nms(bboxes, score_threshold, iou_threshold, sigma=0.3):
    """
    :param bboxes:
    假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    :return: best_bboxes
github    假设NMS后剩下N个bbox，那么best_bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    """
    classes_in_img = list(set(bboxes[:, 5].astype(np.int32)))
    best_bboxes = []
    scale_factor = cfg.SCALE_FACTOR
    for cls in classes_in_img:
        cls_mask = (bboxes[:, 5].astype(np.int32) == cls)
        cls_bboxes = bboxes[cls_mask]
        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 4])#取分数最大的
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])
            #####################################
            iou = iou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            ####################################
            #scale_factor = 1.0 * best_bbox[..., 2:3] * best_bbox[..., 3:4] / (cfg.TEST["TEST_IMG_SIZE"] ** 2)

            method = cfg.TEST["NMS_METHODS"]
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'NMS':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            elif method == 'SOFT_NMS':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            elif method == 'NMS_DIOU':
                diou = diou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                iou_mask = diou > iou_threshold
                weight[iou_mask] = 0.0
            #elif method == 'NMS_DIOU_SCALE':
                #iou_mask = scale_factor-(scale_factor-1.0)*diou > iou_threshold
                #weight[iou_mask] = 0.0

            cls_bboxes[:, 4] = cls_bboxes[:, 4] * weight
            score_mask = cls_bboxes[:, 4] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    return np.array(best_bboxes)


def nms_glid(bboxes, score_threshold, iou_threshold, sigma=0.3):
    """
    :param bboxes:
    假设有N个bbox的score大于score_threshold，那么bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    :return: best_bboxes
    假设NMS后剩下N个bbox，那么best_bboxes的shape为(N, 6)，存储格式为(xmin, ymin, xmax, ymax, score, class)
    其中(xmin, ymin, xmax, ymax)的大小都是相对于输入原图的，score = conf * prob，class是bbox所属类别的索引号
    """
    ######[coors(0:4), coors_rota(4:8), scores[:, np.newaxis](12), classes[:, np.newaxis]](13)
    classes_in_img = list(set(bboxes[:, 9].astype(np.int32)))
    best_bboxes = []
    scale_factor = cfg.SCALE_FACTOR
    for cls in classes_in_img:
        cls_mask = (bboxes[:, 9].astype(np.int32) == cls)
        cls_bboxes = bboxes[cls_mask]

        while len(cls_bboxes) > 0:
            max_ind = np.argmax(cls_bboxes[:, 8])#取分数最大的
            best_bbox = cls_bboxes[max_ind]
            best_bboxes.append(best_bbox)
            cls_bboxes = np.concatenate([cls_bboxes[: max_ind], cls_bboxes[max_ind + 1:]])

            xmin = best_bbox[np.newaxis, 0:1]
            ymin = best_bbox[np.newaxis, 1:2]
            xmax = best_bbox[np.newaxis, 2:3]
            ymax = best_bbox[np.newaxis, 3:4]
            a1 = best_bbox[np.newaxis, 4:5]
            a2 = best_bbox[np.newaxis, 5:6]
            a3 = best_bbox[np.newaxis, 5:6]
            a4 = best_bbox[np.newaxis, 6:7]
            x1 = a1*(xmax-xmin)+xmin
            y1 = ymin
            x2 = a2*(ymax-ymin)+ymin
            y2 = xmax
            x3 = xmax-a3*(xmax-xmin)
            y3 = ymax
            x4 = xmin
            y4 = ymax-a4*(ymax-ymin)
            best_bbox_r = np.concatenate((x1,y1,x2,y2,x3,y3,x4,y4),axis=-1)

            xminl = cls_bboxes[:, 0:1]
            yminl = cls_bboxes[:, 1:2]
            xmaxl = cls_bboxes[:, 2:3]
            ymaxl = cls_bboxes[:, 3:4]
            a1l = cls_bboxes[:, 4:5]
            a2l = cls_bboxes[:, 5:6]
            a3l = cls_bboxes[:, 5:6]
            a4l = cls_bboxes[:, 6:7]
            x1l = a1l*(xmaxl-xminl)+xminl
            y1l = yminl
            x2l = a2l*(ymaxl-yminl)+yminl
            y2l = xmaxl
            x3l = xmaxl-a3l*(xmaxl-xminl)
            y3l = ymaxl
            x4l = xminl
            y4l = ymaxl-a4l*(ymaxl-yminl)
            cls_bboxes_r = np.concatenate((x1l,y1l,x2l,y2l,x3l,y3l,x4l,y4l),axis=-1)
            #print(cls_bboxes_r.shape)
            iou = polygen_iou_xy4_numpy(best_bbox_r[np.newaxis, :8], cls_bboxes_r[:, :8])
            #print(np.maximum(iou,0.7))
            #iou = iou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
            ####################################
            #scale_factor = 1.0 * best_bbox[..., 2:3] * best_bbox[..., 3:4] / (cfg.TEST["TEST_IMG_SIZE"] ** 2)
            method = cfg.TEST["NMS_METHODS"]
            weight = np.ones((len(iou),), dtype=np.float32)
            if method == 'NMS':
                iou_mask = iou > iou_threshold
                weight[iou_mask] = 0.0
            elif method == 'SOFT_NMS':
                weight = np.exp(-(1.0 * iou ** 2 / sigma))
            elif method == 'NMS_DIOU':
                diou = diou_xyxy_numpy(best_bbox[np.newaxis, :4], cls_bboxes[:, :4])
                iou_mask = diou > iou_threshold
                weight[iou_mask] = 0.0
            #elif method == 'NMS_DIOU_SCALE':
                #iou_mask = scale_factor-(scale_factor-1.0)*diou > iou_threshold
                #weight[iou_mask] = 0.0

            cls_bboxes[:, 8] = cls_bboxes[:, 8] * weight
            score_mask = cls_bboxes[:, 8] > score_threshold
            cls_bboxes = cls_bboxes[score_mask]
    return np.array(best_bboxes)

from DOTA_devkit import polyiou

def py_cpu_nms_poly_fast(dets, scores, thresh):
    """
    任意四点poly nms.取出nms后的边框的索引
    @param dets: shape(detection_num, [poly]) 原始图像中的检测出的目标数量
    @param scores: shape(detection_num, 1)
    @param thresh:
    @return:
            keep: 经nms后的目标边框的索引
    """
    obbs = dets[:, 0:-1]  # (num, [poly])
    x1 = np.min(obbs[:, 0::2], axis=1)  # (num, 1)
    y1 = np.min(obbs[:, 1::2], axis=1)  # (num, 1)
    x2 = np.max(obbs[:, 0::2], axis=1)  # (num, 1)
    y2 = np.max(obbs[:, 1::2], axis=1)  # (num, 1)
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)  # (num, 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble(
            [dets[i][0], dets[i][1], dets[i][2], dets[i][3], dets[i][4], dets[i][5], dets[i][6], dets[i][7]]
        )
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]  # argsort将元素小到大排列 返回索引值 [::-1]即从后向前取元素

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]  # 取出当前剩余置信度最大的目标边框的索引
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pass
                # pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep