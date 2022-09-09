import os
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset

import config.config as cfg
import dataloadR.augmentations as DataAug

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class Construct_Dataset(Dataset):
    def __init__(self, anno_file_name, img_size=int(cfg.TRAIN["TRAIN_IMG_SIZE"])):
        self.img_size = img_size
        self.num_classes = len(cfg.DATA["CLASSES"])
        self.stride = [8, 16, 32]
        self.thresh_gh = 0.05
        self.__annotations = self.__load_annotations(anno_file_name)

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            item, self.img_size = item
        else:
            item, self.img_size = item, self.img_size

        img_org, bboxes_org = self.__parse_annotation(self.__annotations[item])
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.__annotations) - 1)
        img_mix, bboxes_mix = self.__parse_annotation(self.__annotations[item_mix])
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = DataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix
        label_sbbox, label_mbbox, label_lbbox = self.__creat_label(bboxes)

        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()

        return img, label_sbbox, label_mbbox, label_lbbox

    def __load_annotations(self, anno_name):
        anno_path = os.path.join(cfg.PROJECT_PATH, 'dataR', anno_name + ".txt")
        with open(anno_path, 'r') as f:
            annotations = list(filter(lambda x: len(x) > 0, f.readlines()))
        assert len(annotations) > 0, "No images found in {}".format(anno_path)
        return annotations

    def __parse_annotation(self, annotation):
        anno = annotation.strip().split(' ')
        img_path = anno[0]
        img = cv2.imread(img_path)
        assert img is not None, 'File Not Found ' + img_path
        bboxes = np.array([list(map(float, box.split(','))) for box in anno[1:]])
        img, bboxes = DataAug.RandomVerticalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomHorizontalFilp()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.HSV()(np.copy(img), np.copy(bboxes))
        # img, bboxes = DataAug.Blur()(np.copy(img), np.copy(bboxes))
        # img, bboxes = DataAug.Gamma()(np.copy(img), np.copy(bboxes))
        # img, bboxes = DataAug.Noise()(np.copy(img), np.copy(bboxes))
        # img, bboxes = DataAug.Contrast()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomCrop()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.RandomAffine()(np.copy(img), np.copy(bboxes))
        img, bboxes = DataAug.Resize((self.img_size, self.img_size), True)(np.copy(img), np.copy(bboxes))
        return img, bboxes

    def generate_label(self, k, gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                       ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                       gt_label, class_id):
        ws = self.img_size // self.stride[k]
        hs = self.img_size // self.stride[k]
        grid_x = int(c_x_r // self.stride[k])
        grid_y = int(c_y_r // self.stride[k])
        r_w = int(len_w / self.stride[k] * 0.5 + 1e-16)
        r_h = int(len_h / self.stride[k] * 0.5 + 1e-16)
        sub_xmin = max(grid_x - r_w - 1, 0)
        sub_xmax = min(grid_x + r_w + 1, ws - 1)
        sub_ymin = max(grid_y - r_h - 1, 0)
        sub_ymax = min(grid_y + r_h + 1, hs - 1)

        sub_xmin1 = max(grid_x - int(box_w // self.stride[k]) - 1, 0)
        sub_xmax1 = min(grid_x + int(box_w // self.stride[k]) + 1, ws - 1)
        sub_ymin1 = max(grid_y - int(box_h // self.stride[k]) - 1, 0)
        sub_ymax1 = min(grid_y + int(box_h // self.stride[k]) + 1, hs - 1)
        gt_tensor_oval_1 = np.zeros([hs, ws, 1])
        for i in range(sub_xmin1, sub_xmax1 + 1):
            for j in range(sub_ymin1, sub_ymax1 + 1):
                l1 = j - sub_ymin
                l2 = sub_xmax - i
                l3 = sub_ymax - j
                l4 = i - sub_xmin
                if min(l1, l2, l3, l4) > 0:
                    gt_tensor_oval_1[j, i, 0:1] = np.sqrt((min(l1, l3) / max(l1, l3)) * (min(l2, l4) / max(l2, l4)))
        sub_tensor = gt_tensor_oval_1[:, :, :]
        M = cv2.getRotationMatrix2D((grid_x, grid_y), angle, 1)
        res = cv2.warpAffine(sub_tensor, M, (ws, hs), cv2.INTER_CUBIC)
        if res.max() != 0:
            res = res / res.max()
        res = (res > self.thresh_gh) * res
        for i in range(sub_xmin1, sub_xmax1 + 1):
            for j in range(sub_ymin1, sub_ymax1 + 1):
                ori_gh = np.max(gt_tensor[k][j, i, 16:], axis=-1)
                maxv = max(res[j, i], ori_gh)
                l1 = j - sub_ymin
                l2 = sub_xmax - i
                l3 = sub_ymax - j
                l4 = i - sub_xmin
                if (ori_gh <= maxv) and maxv > self.thresh_gh:
                    gt_tensor[k][j, i, 0:8] = np.array([c_x, c_y, box_w, box_h, l1, l2, l3, l4])
                    gt_tensor[k][j, i, 8:12] = np.array([a1, a2, a3, a4])
                    gt_tensor[k][j, i, 12] = gt_label[13]
                    gt_tensor[k][j, i, 13] = 1.0
                    gt_tensor[k][j, i, 16 + class_id:16 + class_id + 1] = maxv
                    gt_tensor[k][j, i, 15] = gt_label[14]
        gt_tensor[k][:, :, 14] = gt_label[15]

    def __creat_label(self, label_lists=[]):
        self.gt_tensor = [np.zeros((int(self.img_size // self.stride[i]), int(self.img_size // self.stride[i]),
                                    self.num_classes + 16)) for i in range(3)]
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
            class_id = int(gt_label[4])
            len_w = (np.sqrt((bbox_obb[5] - bbox_obb[3]) ** 2 + (bbox_obb[2] - bbox_obb[4]) ** 2)
                     + np.sqrt((bbox_obb[7] - bbox_obb[1]) ** 2 + (bbox_obb[0] - bbox_obb[6]) ** 2)) / 2
            len_h = (np.sqrt((bbox_obb[1] - bbox_obb[3]) ** 2 + (bbox_obb[2] - bbox_obb[0]) ** 2) +
                     np.sqrt((bbox_obb[4] - bbox_obb[6]) ** 2 + (bbox_obb[5] - bbox_obb[7]) ** 2)) / 2
            c_x_r = (bbox_obb[0] + bbox_obb[2] + bbox_obb[4] + bbox_obb[6]) / 4
            c_y_r = (bbox_obb[1] + bbox_obb[3] + bbox_obb[5] + bbox_obb[7]) / 4
            angle = 180 - gt_label[14]
            if len_w < len_h:
                len_w, len_h = len_h, len_w
            length = max(box_w, box_h)
            if max(box_w, box_h) > 10 or (box_w * box_w) > 80:
                if length <= 128:
                    self.generate_label(0, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)

                if length > 128 and 256:
                    self.generate_label(1, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)

                if length > 256:
                    self.generate_label(2, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)

        label_sbbox, label_mbbox, label_lbbox = self.gt_tensor
        return label_sbbox, label_mbbox, label_lbbox

