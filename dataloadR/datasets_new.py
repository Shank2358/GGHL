# coding=utf-8
import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import numpy as np
import cv2
import random
import math
import torch
from torch.utils.data import Dataset

import config.cfg_npmmr as cfg
import dataloadR.augmentations as DataAug
import utils.utils_basic as tools
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker


class Construct_Dataset(Dataset):
    def __init__(self, anno_file_type, img_size=int(cfg.TRAIN["TRAIN_IMG_SIZE"])):
        self.img_size = img_size  # For Multi-training
        self.classes = cfg.DATA["CLASSES"]
        self.num_classes = len(self.classes)
        self.class_to_id = dict(zip(self.classes, range(self.num_classes)))
        self.__annotations = self.__load_annotations(anno_file_type)
        self.__IOU_thresh = 0.3
        self.thresh_gh = 0.05

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):
        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.__annotations) - 1)
        img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = DataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix
        label_sbbox, label_mbbox, label_lbbox = self.__creat_label(bboxes)

        '''
        img = np.uint8(np.transpose(img, (1, 2, 0)) * 255)
        plt.figure("img")  # 图像窗口名称
        plt.imshow(img)

        # (label_sbbox[:, :, 2:3] > 0)
        mask_s = np.max(label_sbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_s")  # 图像窗口名称
        plt.imshow(mask_s, cmap='jet')

        mask_s1 = np.max(label_sbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_s1")  # 图像窗口名称
        plt.imshow(mask_s1, cmap='jet')

        imgs = cv2.resize(img, dsize=None, fx=0.125, fy=0.125, interpolation=cv2.INTER_NEAREST)
        mask_s = mask_s * 255
        mask_s = np.uint8(np.concatenate((mask_s, mask_s, mask_s), axis=2))
        mask_s = cv2.applyColorMap(mask_s, cv2.COLORMAP_RAINBOW)
        add_img = cv2.addWeighted(imgs, 0.5, mask_s, 0.5, 0)
        plt.figure("ImageS")  # 图像窗口名称
        plt.imshow(add_img / 255, cmap='jet')
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        mask_m = np.max(label_mbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_m")  # 图像窗口名称
        plt.imshow(mask_m, cmap='jet')

        mask_m1 = np.max(label_mbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_m1")  # 图像窗口名称
        plt.imshow(mask_m1, cmap='jet')

        imgm = cv2.resize(img, dsize=None, fx=1 / 16, fy=1 / 16, interpolation=cv2.INTER_NEAREST)
        mask_m = mask_m * 255
        mask_m = np.uint8(np.concatenate((mask_m, mask_m, mask_m), axis=2))
        mask_m = cv2.applyColorMap(mask_m, cv2.COLORMAP_RAINBOW)
        add_img = cv2.addWeighted(imgm, 0.5, mask_m, 0.5, 0)
        plt.figure("ImageM")  # 图像窗口名称
        plt.imshow(add_img / 255, cmap='jet')
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        mask_l = np.max(label_lbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_l")  # 图像窗口名称
        plt.imshow(mask_l, cmap='jet')

        mask_l1 = np.max(label_lbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_l1")  # 图像窗口名称
        plt.imshow(mask_l1, cmap='jet')

        imgl = cv2.resize(img, dsize=None, fx=1 / 32, fy=1 / 32, interpolation=cv2.INTER_NEAREST)
        mask_l = mask_l * 255
        mask_l = np.uint8(np.concatenate((mask_l, mask_l, mask_l), axis=2))
        mask_l = cv2.applyColorMap(mask_l, cv2.COLORMAP_RAINBOW)
        add_img = cv2.addWeighted(imgl, 0.5, mask_l, 0.5, 0)
        plt.figure("ImageL")  # 图像窗口名称
        plt.imshow(add_img / 255, cmap='jet')
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        plt.show()'''

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()
        return img, label_sbbox, label_mbbox, label_lbbox

    def __load_annotations(self, anno_type):
        assert anno_type in ['train', 'val', 'test']
        anno_path = os.path.join(cfg.PROJECT_PATH, 'dataR', anno_type + ".txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))

        assert len(annotations) > 0, "No images found in {}".format(anno_path)
        return annotations

    def __parse_annotation(self, annotation):
        anno = annotation.strip().split(' ')
        img_path = anno[0]
        img = cv2.imread(img_path)  # H*W*C and C=BGR
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array(
            [list(map(float, box.split(','))) for box in anno[1:]])  ####xmin,ymin,xmax,ymax,c,x1,y1,x2,y2,x3,y3,x4,y4,r
        img, bboxes = DataAug.RandomVerticalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.HSV()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Blur()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Gamma()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Noise()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Contrast()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        return img, bboxes

    def generate(self, c_x, c_y, bbox_obb, len_w, len_h, s, ratio):
        sigma_w = len_w / s / 3 * ratio + 1e-14
        sigma_h = len_h / s / 3 * ratio + 1e-14
        sigma_w_r = len_w / s / 3 * 0.5 + 1e-14
        sigma_h_r = len_h / s / 3 * 0.5 + 1e-14
        c_x_r = (bbox_obb[0] + bbox_obb[2] + bbox_obb[4] + bbox_obb[6]) / 4
        c_y_r = (bbox_obb[1] + bbox_obb[3] + bbox_obb[5] + bbox_obb[7]) / 4
        grid_x = int(c_x_r / s)
        grid_y = int(c_y_r / s)
        return grid_x, grid_y, sigma_w, sigma_h, sigma_w_r, sigma_h_r

    def sigmoid(self, x):
        return 1 / (1 + np.exp(-x))

    def __creat_label(self, label_lists=[]):
        stride = np.array(cfg.MODEL["STRIDES"])
        gt_tensor = [np.zeros((int(self.img_size // stride[i]), int(self.img_size // stride[i]),
                               6 + 5 + self.num_classes + 4 + 1)) for i in range(3)]
        for gt_label in label_lists:
            bbox_xyxy = gt_label[:4]
            bbox_obb = gt_label[5:13]
            xmin, ymin, xmax, ymax = bbox_xyxy
            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            c_x = (xmax + xmin) / 2
            c_y = (ymax + ymin) / 2
            if gt_label[13] > 0.9:
                a1 = a2 = a3 = a4 = 0
            else:
                a1 = (bbox_obb[0] - bbox_xyxy[0]) / box_w
                a2 = (bbox_obb[3] - bbox_xyxy[1]) / box_h
                a3 = (bbox_xyxy[2] - bbox_obb[4]) / box_w
                a4 = (bbox_xyxy[3] - bbox_obb[7]) / box_h
            bbox_class_ind = int(gt_label[4])
            len_w = (np.sqrt((bbox_obb[5] - bbox_obb[3]) ** 2 + (bbox_obb[2] - bbox_obb[4]) ** 2)
                     + np.sqrt((bbox_obb[7] - bbox_obb[1]) ** 2 + (bbox_obb[0] - bbox_obb[6]) ** 2)) / 2
            len_h = (np.sqrt((bbox_obb[1] - bbox_obb[3]) ** 2 + (bbox_obb[2] - bbox_obb[0]) ** 2) +
                     np.sqrt((bbox_obb[4] - bbox_obb[6]) ** 2 + (bbox_obb[5] - bbox_obb[7]) ** 2)) / 2
            length = max(len_w, len_h)
            c_x_r = (bbox_obb[0] + bbox_obb[2] + bbox_obb[4] + bbox_obb[6]) / 4
            c_y_r = (bbox_obb[1] + bbox_obb[3] + bbox_obb[5] + bbox_obb[7]) / 4
            angle = (np.arctan((1 - a2) / (a3 + 1e-16)) + np.arctan((1 - a4) / (a1 + 1e-16))) / 2
            ratio = (1-self.__IOU_thresh)
            if box_w > 1 and box_h > 1:
                if length <= 3*(stride[0]*2)/ratio:
                    ws = self.img_size // stride[0]
                    hs = self.img_size // stride[0]
                    grid_x = int(c_x_r / stride[0])
                    grid_y = int(c_y_r / stride[0])
                    r_w = len_w / stride[0] * 0.5 + 1e-16
                    r_h = len_h / stride[0] * 0.5 + 1e-16
                    gt_tensor_oval = np.zeros([hs, ws, self.num_classes])
                    r_w_max = int(np.clip(box_w / stride[0] / 2, 1, np.inf))
                    r_h_max = int(np.clip(box_h / stride[0] / 2, 1, np.inf))
                    sub_xmin = max(grid_x - r_w_max, 0)
                    sub_xmax = min(grid_x + r_w_max + 1, ws - 1)
                    sub_ymin = max(grid_y - r_h_max, 0)
                    sub_ymax = min(grid_y + r_h_max + 1, hs - 1)
                    gt_tensor_oval_1 = np.zeros([hs, ws, 1])
                    gt_tensor_oval_mask = np.zeros([hs, ws, 1])
                    for i in range(grid_x - r_w_max, grid_x + r_w_max + 1):
                        for j in range(grid_y - r_h_max, grid_y + r_h_max + 1):
                            if i < ws and j < hs:
                                ax = np.array([[i - grid_x, j - grid_y]]).transpose()
                                R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                                Eig = np.array([[2/r_w, 0], [0, 2/r_h]])
                                axnew = np.dot(np.dot(Eig, R), ax)
                                v = np.exp(- (axnew[0, 0] ** 2 + axnew[1, 0] ** 2) / 2) #/ (2 * np.pi)
                                pre_v_oval = gt_tensor_oval_1[j, i, 0:1]
                                gt_tensor_oval_1[j, i, 0:1] = max(v, pre_v_oval)

                                Eig_mask = np.array([[2 / r_w, 0], [0, 2 / r_h]]) / ratio
                                axnew_mask = np.dot(np.dot(Eig_mask, R), ax)
                                v_mask = np.exp(- (axnew_mask[0, 0] ** 2 + axnew_mask[1, 0] ** 2) / 2) / (2 * np.pi)
                                pre_v_oval_mask = gt_tensor_oval_mask[j, i, 0:1]
                                gt_tensor_oval_mask[j, i, 0:1] = max(v_mask, pre_v_oval_mask)
                    sub_tensor = gt_tensor_oval_1[sub_ymin:sub_ymax, sub_xmin:sub_xmax, :]
                    res = sub_tensor
                    if res.max() != 0:
                        res = res/res.max()
                    res = (res > self.thresh_gh) * res
                    sub_tensor_mask = gt_tensor_oval_mask[sub_ymin:sub_ymax, sub_xmin:sub_xmax, :]
                    if sub_tensor_mask.max() != 0:
                        sub_tensor_mask = sub_tensor_mask/sub_tensor_mask.max()
                    res = res * (sub_tensor_mask > self.thresh_gh)
                    gt_tensor_oval[sub_ymin:sub_ymax, sub_xmin:sub_xmax, bbox_class_ind:bbox_class_ind + 1] = res
                    for i in range(sub_xmin, sub_xmax):
                        for j in range(sub_ymin, sub_ymax):
                            l1 = (j * stride[0] + stride[0] / 2) - ymin
                            l2 = xmax - (i * stride[0] + stride[0] / 2)
                            l3 = ymax - (j * stride[0] + stride[0] / 2)
                            l4 = (i * stride[0] + stride[0] / 2) - xmin
                            ori_gh = np.max(gt_tensor[0][j, i, 16:], axis=-1)
                            new_gh = np.max(gt_tensor_oval[j, i, :], axis=-1)
                            if (ori_gh < new_gh) and min(l1, l2, l3, l4) > 0:
                                gt_tensor[0][j, i, 0:8] = np.array([c_x, c_y, box_w, box_h, l1, l2, l3, l4])
                                gt_tensor[0][j, i, 8:12] = np.array([a1, a2, a3, a4])
                                gt_tensor[0][j, i, 12] = gt_label[13]
                                gt_tensor[0][j, i, 13] = 1.0
                                gt_tensor[0][j, i, 16:] = gt_tensor_oval[j, i, :]
                                area = np.sum(res != 0)
                                gt_tensor[0][j, i, 15] = 2* np.log(2)/np.log(np.sqrt(area)+1)
                    gt_tensor[0][:, :, 14] = gt_label[14]

                if length > 3*(stride[0]*2)/ratio and length <= 3*(stride[2]*2)/ratio:
                    ws = self.img_size // stride[1]
                    hs = self.img_size // stride[1]
                    grid_x = int(c_x_r / stride[1])
                    grid_y = int(c_y_r / stride[1])
                    r_w = len_w / stride[1] * 0.5 + 1e-16
                    r_h = len_h / stride[1] * 0.5 + 1e-16
                    gt_tensor_oval = np.zeros([hs, ws, self.num_classes])
                    r_w_max = int(np.clip(box_w / stride[1] / 2, 1, np.inf))
                    r_h_max = int(np.clip(box_h / stride[1] / 2, 1, np.inf))
                    sub_xmin = max(grid_x - r_w_max, 0)
                    sub_xmax = min(grid_x + r_w_max + 1, ws - 1)
                    sub_ymin = max(grid_y - r_h_max, 0)
                    sub_ymax = min(grid_y + r_h_max + 1, hs - 1)
                    gt_tensor_oval_1 = np.zeros([hs, ws, 1])
                    gt_tensor_oval_mask = np.zeros([hs, ws, 1])
                    for i in range(grid_x - r_w_max, grid_x + r_w_max + 1):
                        for j in range(grid_y - r_h_max, grid_y + r_h_max + 1):
                            if i < ws and j < hs:
                                ax = np.array([[i - grid_x, j - grid_y]]).transpose()
                                R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                                Eig = np.array([[2 / r_w, 0], [0, 2 / r_h]])
                                axnew = np.dot(np.dot(Eig, R), ax)
                                v = np.exp(- (axnew[0, 0] ** 2 + axnew[1, 0] ** 2) / 2) / (2 * np.pi)
                                pre_v_oval = gt_tensor_oval_1[j, i, 0:1]
                                gt_tensor_oval_1[j, i, 0:1] = max(v, pre_v_oval)

                                Eig_mask = np.array([[2 / r_w, 0], [0, 2 / r_h]]) / ratio
                                axnew_mask = np.dot(np.dot(Eig_mask, R), ax)
                                v_mask = np.exp(- (axnew_mask[0, 0] ** 2 + axnew_mask[1, 0] ** 2) / 2) / (2 * np.pi)
                                pre_v_oval_mask = gt_tensor_oval_mask[j, i, 0:1]
                                gt_tensor_oval_mask[j, i, 0:1] = max(v_mask, pre_v_oval_mask)
                    sub_tensor = gt_tensor_oval_1[sub_ymin:sub_ymax, sub_xmin:sub_xmax, :]
                    res = sub_tensor
                    if res.max() != 0:
                        res = res / res.max()
                    res = (res > self.thresh_gh) * res
                    sub_tensor_mask = gt_tensor_oval_mask[sub_ymin:sub_ymax, sub_xmin:sub_xmax, :]
                    if sub_tensor_mask.max() != 0:
                        sub_tensor_mask = sub_tensor_mask / sub_tensor_mask.max()
                    res = res * (sub_tensor_mask > self.thresh_gh)
                    gt_tensor_oval[sub_ymin:sub_ymax, sub_xmin:sub_xmax, bbox_class_ind:bbox_class_ind + 1] = res
                    for i in range(sub_xmin, sub_xmax):
                        for j in range(sub_ymin, sub_ymax):
                            l1 = (j * stride[1] + stride[1] / 2) - ymin
                            l2 = xmax - (i * stride[1] + stride[1] / 2)
                            l3 = ymax - (j * stride[1] + stride[1] / 2)
                            l4 = (i * stride[1] + stride[1] / 2) - xmin
                            ori_gh = np.max(gt_tensor[1][j, i, 16:], axis=-1)
                            new_gh = np.max(gt_tensor_oval[j, i, :], axis=-1)
                            if (ori_gh < new_gh) and min(l1, l2, l3, l4) > 0:
                                gt_tensor[1][j, i, 0:8] = np.array([c_x, c_y, box_w, box_h, l1, l2, l3, l4])
                                gt_tensor[1][j, i, 8:12] = np.array([a1, a2, a3, a4])
                                gt_tensor[1][j, i, 12] = gt_label[13]
                                gt_tensor[1][j, i, 13] = 1.0
                                gt_tensor[1][j, i, 16:] = gt_tensor_oval[j, i, :]
                                area = np.sum(res != 0)
                                gt_tensor[1][j, i, 15] = 2*np.log(2)/np.log(np.sqrt(area)+1)
                    gt_tensor[1][:, :, 14] = gt_label[14]

                if length > 3*(stride[2]*2)/ratio:
                    ws = self.img_size // stride[2]
                    hs = self.img_size // stride[2]
                    grid_x = int(c_x_r / stride[2])
                    grid_y = int(c_y_r / stride[2])
                    r_w = len_w / stride[2] * 0.5 + 1e-16
                    r_h = len_h / stride[2] * 0.5 + 1e-16
                    gt_tensor_oval = np.zeros([hs, ws, self.num_classes])
                    r_w_max = int(np.clip(box_w / stride[2] / 2, 1, np.inf))
                    r_h_max = int(np.clip(box_h / stride[2] / 2, 1, np.inf))
                    sub_xmin = max(grid_x - r_w_max, 0)
                    sub_xmax = min(grid_x + r_w_max + 1, ws - 1)
                    sub_ymin = max(grid_y - r_h_max, 0)
                    sub_ymax = min(grid_y + r_h_max + 1, hs - 1)
                    gt_tensor_oval_1 = np.zeros([hs, ws, 1])
                    gt_tensor_oval_mask = np.zeros([hs, ws, 1])
                    for i in range(grid_x - r_w_max, grid_x + r_w_max + 1):
                        for j in range(grid_y - r_h_max, grid_y + r_h_max + 1):
                            if i < ws and j < hs:
                                ax = np.array([[i - grid_x, j - grid_y]]).transpose()
                                R = np.array([[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]])
                                Eig = np.array([[2 / r_w, 0], [0, 2 / r_h]])
                                axnew = np.dot(np.dot(Eig, R), ax)
                                v = np.exp(- (axnew[0, 0] ** 2 + axnew[1, 0] ** 2) / 2) / (2 * np.pi)
                                pre_v_oval = gt_tensor_oval_1[j, i, 0:1]
                                gt_tensor_oval_1[j, i, 0:1] = max(v, pre_v_oval)

                                Eig_mask = np.array([[2 / r_w, 0], [0, 2 / r_h]]) / ratio
                                axnew_mask = np.dot(np.dot(Eig_mask, R), ax)
                                v_mask = np.exp(- (axnew_mask[0, 0] ** 2 + axnew_mask[1, 0] ** 2) / 2) / (2 * np.pi)
                                pre_v_oval_mask = gt_tensor_oval_mask[j, i, 0:1]
                                gt_tensor_oval_mask[j, i, 0:1] = max(v_mask, pre_v_oval_mask)

                    sub_tensor = gt_tensor_oval_1[sub_ymin:sub_ymax, sub_xmin:sub_xmax, :]
                    res = sub_tensor
                    if res.max() != 0:
                        res = res / res.max()
                    res = (res > self.thresh_gh) * res
                    sub_tensor_mask = gt_tensor_oval_mask[sub_ymin:sub_ymax, sub_xmin:sub_xmax, :]
                    if sub_tensor_mask.max() != 0:
                        sub_tensor_mask = sub_tensor_mask / sub_tensor_mask.max()
                    res = res * (sub_tensor_mask > self.thresh_gh)
                    gt_tensor_oval[sub_ymin:sub_ymax, sub_xmin:sub_xmax, bbox_class_ind:bbox_class_ind + 1] = res
                    for i in range(sub_xmin, sub_xmax):
                        for j in range(sub_ymin, sub_ymax):
                            l1 = (j * stride[2] + stride[2] / 2) - ymin
                            l2 = xmax - (i * stride[2] + stride[2] / 2)
                            l3 = ymax - (j * stride[2] + stride[2] / 2)
                            l4 = (i * stride[2] + stride[2] / 2) - xmin
                            ori_gh = np.max(gt_tensor[2][j, i, 16:], axis=-1)
                            new_gh = np.max(gt_tensor_oval[j, i, :], axis=-1)
                            if (ori_gh < new_gh) and min(l1, l2, l3, l4) > 0:
                                gt_tensor[2][j, i, 0:8] = np.array([c_x, c_y, box_w, box_h, l1, l2, l3, l4])
                                gt_tensor[2][j, i, 8:12] = np.array([a1, a2, a3, a4])  # a
                                gt_tensor[2][j, i, 12] = gt_label[13]
                                gt_tensor[2][j, i, 13] = 1.0
                                gt_tensor[2][j, i, 16:] = gt_tensor_oval[j, i, :]
                                area = np.sum(res != 0)
                                gt_tensor[2][j, i, 15] = 2*np.log(2)/np.log(np.sqrt(area)+1)
                    gt_tensor[2][:, :, 14] = gt_label[14]

        label_sbbox, label_mbbox, label_lbbox = gt_tensor
        label_sbbox[:, :, 16:] = label_sbbox[:, :, 13:14] * label_sbbox[:, :, 16:]
        label_mbbox[:, :, 16:] = label_mbbox[:, :, 13:14] * label_mbbox[:, :, 16:]
        label_lbbox[:, :, 16:] = label_lbbox[:, :, 13:14] * label_lbbox[:, :, 16:]
        return label_sbbox, label_mbbox, label_lbbox


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_dataset = Construct_Dataset(anno_file_type="train", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    for i, (imgs, label_sbbox, label_mbbox, label_lbbox) in enumerate(train_dataloader):
        continue