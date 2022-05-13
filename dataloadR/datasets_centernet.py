# coding=utf-8
import os
import sys
sys.path.append("..")
sys.path.append("../utils")
import numpy as np
import cv2
import random
import torch
from torch.utils.data import Dataset
import config.config as cfg
import dataloadR.augmentations as DataAug

class Construct_Dataset(Dataset):
    def __init__(self, anno_file_name, img_size=int(cfg.TRAIN["TRAIN_IMG_SIZE"]), load_RAM=False):
        self.img_size = img_size
        self.num_classes = len(cfg.DATA["CLASSES"])
        self.stride = [8, 16, 32]
        self.IOU_thresh = 0.3
        self.thresh_gh = 0.05
        self.__annotations = self.__load_annotations(anno_file_name)
        self.img_RAM = [None] * len(self.__annotations)
        if load_RAM:
            for i, img_path in enumerate(self.__annotations):
                self.img_RAM[i] = cv2.imread(img_path[0])

    def __len__(self):
        return len(self.__annotations)

    def __getitem__(self, item):
        if type(item) == list or type(item) == tuple:
            item, self.img_size = item
        else:
            item, self.img_size = item, self.img_size

        img_org, bboxes_org = self.__parse_annotation(item)
        img_org = img_org.transpose(2, 0, 1)  # HWC->CHW

        item_mix = random.randint(0, len(self.__annotations) - 1)
        img_mix, bboxes_mix = self.__parse_annotation(item_mix)
        img_mix = img_mix.transpose(2, 0, 1)

        img, bboxes = DataAug.Mixup()(img_org, bboxes_org, img_mix, bboxes_mix)
        del img_org, bboxes_org, img_mix, bboxes_mix
        label_bbox = self.__creat_label(bboxes)

        img = torch.from_numpy(img).float()
        label_bbox = torch.from_numpy(label_bbox).float()

        return img, label_bbox

    def __load_annotations(self, anno_name):
        anno_path = os.path.join(cfg.PROJECT_PATH, 'dataR', anno_name + ".txt")
        with open(anno_path, 'r') as f:
            annotation = filter(lambda x: len(x) > 0, f.readlines())
            annotations = list(annotation)
        assert len(annotations) > 0, "No images found in {}".format(anno_path)
        annotations = [x.strip().split(' ') for x in annotations]
        return annotations

    def __parse_annotation(self, index):
        anno = self.__annotations[index]
        if self.img_RAM[index] is not None:
            img = self.img_RAM[index]
        else:
            img = cv2.imread(anno[0])
        assert img is not None, 'File Not Found ' + anno[0]
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

    def gaussian_radius(self, det_size, min_overlap=0.7):
        width, height = det_size
        a1 = 1
        b1 = (height + width)
        c1 = width * height * (1 - min_overlap) / (1 + min_overlap)
        sq1 = np.sqrt(b1 ** 2 - 4 * a1 * c1)
        r1 = (b1 + sq1) / (2 * a1)  # (2*a1)

        a2 = 4
        b2 = 2 * (height + width)
        c2 = (1 - min_overlap) * width * height
        sq2 = np.sqrt(b2 ** 2 - 4 * a2 * c2)
        r2 = (b2 + sq2) / (2 * a2)  # (2*a2)

        a3 = 4 * min_overlap
        b3 = -2 * min_overlap * (height + width)
        c3 = (min_overlap - 1) * width * height
        sq3 = np.sqrt(b3 ** 2 - 4 * a3 * c3)
        r3 = (b3 + sq3) / (2 * a3)  # (2*a3)
        return min(r1, r2, r3)

    def generate_xywh(self, box_w, box_h, gt_label, s):
        xmin, ymin, xmax, ymax = gt_label
        c_x = (xmax + xmin) / 2
        c_y = (ymax + ymin) / 2
        grid_x = int(c_x / s)
        grid_y = int(c_y / s)
        c_x_s = c_x / s
        c_y_s = c_y / s
        box_w_s = box_w / s
        box_h_s = box_h / s

        # The original centernet's heatmap (Gaussian circle)
        #r = self.gaussian_radius([box_w / s, box_h / s])
        #sigma_w = sigma_h = r / 3

        #The TTFNet's heatmap (Gaussian ellipse without rotation)
        #ratio = np.sqrt(2)/2*(1-np.sqrt(0.7))
        sigma_w = 0.1155 * box_w_s #/ 6 * ratio
        sigma_h = 0.1155 * box_h_s #/ 6 * ratio

        # compute the (x, y, w, h) for the corresponding grid cell
        tx = c_x_s - grid_x
        ty = c_y_s - grid_y
        tw = np.log(box_w_s)
        th = np.log(box_h_s)

        return grid_x, grid_y, c_x, c_y, box_w, box_h, sigma_w, sigma_h, tx, ty, tw, th

    def __creat_label(self, label_lists=[]):
        stride = 4
        ws = self.img_size // stride
        hs = self.img_size // stride
        gt_tensor = np.zeros([ws, hs, 16 + self.num_classes])
        for gt_label in label_lists:
            bbox_xyxy = gt_label[:4]
            xmin, ymin, xmax, ymax = bbox_xyxy
            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            bbox_obb = gt_label[5:13]
            if box_w > 4 and box_h > 4:
                if gt_label[13] > 0.9:
                    a1 = a2 = a3 = a4 = 0
                else:
                    a1 = (bbox_obb[0] - bbox_xyxy[0]) / box_w
                    a2 = (bbox_obb[3] - bbox_xyxy[1]) / box_h
                    a3 = (bbox_xyxy[2] - bbox_obb[4]) / box_w
                    a4 = (bbox_xyxy[3] - bbox_obb[7]) / box_h
                angle = gt_label[14] * np.pi / 180
                result = self.generate_xywh(box_w, box_h, bbox_xyxy, stride)
                grid_x, grid_y, c_x, c_y, box_w, box_h, sigma_w, sigma_h, tx, ty, tw, th = result
                gt_tensor[grid_y, grid_x, 0:8] = np.array([c_x, c_y, box_w, box_h, tx, ty, tw, th])
                gt_tensor[grid_y, grid_x, 8:12] = np.array([a1, a2, a3, a4])
                gt_tensor[grid_y, grid_x, 12] = gt_label[13]
                gt_tensor[grid_y, grid_x, 13] = 1.0
                gt_tensor[grid_y, grid_x, 14] = gt_label[15] ##mix
                gt_tensor[grid_y, grid_x, 15] = angle
                bbox_class_ind = int(gt_label[4])
                gt_tensor_temp = np.zeros([ws, hs, self.num_classes])
                for i in range(grid_x - 3 * int(sigma_w), grid_x + 3 * int(sigma_w) + 1):
                    for j in range(grid_y - 3 * int(sigma_h), grid_y + 3 * int(sigma_h) + 1):
                        if i < ws and j < hs:
                            v = np.exp(- (i - grid_x) ** 2 / (2 * sigma_w ** 2) - (j - grid_y) ** 2 / (2 * sigma_h ** 2))
                            pre_v = gt_tensor[j, i, 16 + bbox_class_ind]
                            gt_tensor[j, i, 16 + bbox_class_ind] = max(v, pre_v)
                            pre_v_temp = gt_tensor_temp[j, i, bbox_class_ind]
                            gt_tensor_temp[j, i, bbox_class_ind] = max(v, pre_v_temp)
        return gt_tensor

if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_dataset = Construct_Dataset(anno_file_name="train_HRSC2016", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=0, shuffle=False)
    for i, (imgs, label_bbox) in enumerate(train_dataloader):
        continue