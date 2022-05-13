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
        label_sbbox, label_mbbox, label_lbbox = self.__creat_label(bboxes)

        ###################画图的Label assignment visualization
        '''
        img_show = cv2.UMat(img.transpose(1, 2, 0)).get() * 255
        img_show = img_show.astype(np.uint8)
        for anno in bboxes:
            cls = int(anno[4])
            bbox_xyxy = anno[:4]
            xmin, ymin, xmax, ymax = bbox_xyxy
            box_w = (xmax - xmin)
            box_h = (ymax - ymin)
            if (max(box_w, box_h) > 10 or (box_w * box_w) > 80) and box_w > 8 and box_h > 8:
                points = np.array(
                    [[int(anno[5]), int(anno[6])], [int(anno[7]), int(anno[8])], [int(anno[9]), int(anno[10])],
                     [int(anno[11]), int(anno[12])]])
                if cls == 0:
                    color = (171, 26, 25)
                elif cls == 1:
                    color = (108, 93, 200)
                elif cls == 2:
                    color = (189, 95, 79)
                elif cls == 3:
                    color = (240, 131, 131)
                elif cls == 4:
                    color = (250, 18, 250)
                elif cls == 5:
                    color = (255, 224, 5)
                elif cls == 6:
                    # color = (219, 141, 14)
                    color = (5, 224, 255)
                elif cls == 7:
                    color = (235, 225, 144)
                elif cls == 8:
                    color = (49, 205, 49)
                elif cls == 9:
                    color = (36, 138, 38)
                elif cls == 10:
                    color = (24, 232, 128)
                elif cls == 11:
                    color = (188, 249, 250)
                elif cls == 12:
                    # color = (108, 210, 201)
                    color = (255, 21, 64)
                elif cls == 13:
                    color = (51, 157, 255)
                elif cls == 14:
                    color = (197, 197, 196)
                cv2.polylines(img_show, [points], 1, color, 2)
        import matplotlib.pyplot as plt

        plt.figure("Image")  # 图像窗口名称
        plt.imshow(img_show)

        import matplotlib.pyplot as plt
        import matplotlib.ticker as ticker

        img = np.uint8(np.transpose(img, (1, 2, 0)) * 255)
        # plt.figure("img")  # 图像窗口名称
        # plt.imshow(img)
        mask_s = np.max(label_sbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_s")  # 图像窗口名称
        plt.imshow(mask_s, cmap='jet', vmax=1, vmin=0)
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        mask_m = np.max(label_mbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_m")  # 图像窗口名称
        plt.imshow(mask_m, cmap='jet', vmax=1, vmin=0)
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        mask_l = np.max(label_lbbox[:, :, 16:], -1, keepdims=True)
        plt.figure("mask_l")  # 图像窗口名称
        plt.imshow(mask_l, cmap='jet', vmax=1, vmin=0)
        cb = plt.colorbar()
        tick_locator = ticker.MaxNLocator(nbins=5)
        cb.locator = tick_locator
        cb.update_ticks()

        plt.show()'''
        ########################



        img = torch.from_numpy(img).float()
        label_sbbox = torch.from_numpy(label_sbbox).float()
        label_mbbox = torch.from_numpy(label_mbbox).float()
        label_lbbox = torch.from_numpy(label_lbbox).float()

        return img, label_sbbox, label_mbbox, label_lbbox

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
        #img, bboxes = DataAug.Blur()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Gamma()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Noise()(np.copy(img), np.copy(bboxes))
        #img, bboxes = DataAug.Contrast()(np.copy(img), np.copy(bboxes))
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
        r_w = len_w / self.stride[k] * 0.5 + 1e-16
        r_h = len_h / self.stride[k] * 0.5 + 1e-16
        r_w_max = int(np.clip(np.power(box_w / self.stride[k] / 2, 1), 1, 4-k))
        r_h_max = int(np.clip(np.power(box_h / self.stride[k] / 2, 1), 1, 4-k))
        sub_xmin = max(grid_x - r_w_max - 1, 0)
        sub_xmax = min(grid_x + r_w_max + 1, ws - 1)
        sub_ymin = max(grid_y - r_h_max - 1, 0)
        sub_ymax = min(grid_y + r_h_max + 1, hs - 1)
        gt_tensor_oval_1 = np.zeros([hs, ws, 1])
        R = np.array([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
        Eig = np.array([[2 / np.power(r_w, 1-self.IOU_thresh), 0], [0, 2 / np.power(r_h, 1-self.IOU_thresh)]])
        for i in range(sub_xmin, sub_xmax + 1):
            for j in range(sub_ymin, sub_ymax + 1):
                ax = np.array([[i - grid_x, j - grid_y]]).transpose()
                axnew = np.dot(np.dot(Eig, R), ax)
                v = np.exp(- (axnew[0, 0] ** 2 + axnew[1, 0] ** 2) / 2)
                pre_v_oval = gt_tensor_oval_1[j, i, 0:1]
                maxv = max(v, pre_v_oval)
                l1 = (j * self.stride[k] + self.stride[k] / 2) - ymin
                l2 = xmax - (i * self.stride[k] + self.stride[k] / 2)
                l3 = ymax - (j * self.stride[k] + self.stride[k] / 2)
                l4 = (i * self.stride[k] + self.stride[k] / 2) - xmin
                ori_gh = np.max(gt_tensor[k][j, i, 16 + class_id: 16 + class_id + 1], axis=-1)
                if (ori_gh <= maxv) and maxv > self.thresh_gh and min(l1, l2, l3, l4) > 0:
                    gt_tensor[k][j, i, 0:8] = np.array([c_x, c_y, box_w, box_h, l1, l2, l3, l4])
                    gt_tensor[k][j, i, 8:12] = np.array([a1, a2, a3, a4])
                    gt_tensor[k][j, i, 12] = gt_label[13]
                    gt_tensor[k][j, i, 13] = 1.0
                    gt_tensor[k][j, i, 16 + class_id:16 + class_id + 1] = maxv
                    gt_tensor[k][j, i, 15] = np.log(2) / np.log(np.sqrt(r_w_max * r_h_max) + 1)
            gt_tensor[k][:, :, 14] = gt_label[15]

    def __creat_label(self, label_lists=[]):
        self.gt_tensor = [np.zeros((int(self.img_size // self.stride[i]), int(self.img_size // self.stride[i]),
                               self.num_classes + 16)) for i in range(3)]
        ratio = (1 - self.IOU_thresh)
        layer_thresh = [3*(self.stride[0]*2)/ratio, 3*(self.stride[2]*2)/ratio]
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
            angle = gt_label[14] * np.pi / 180
            if len_w < len_h:
                len_w, len_h = len_h, len_w
            length = max(box_w, box_h)
            if max(box_w, box_h) > 10 or (box_w*box_w) > 80:
                if length <= layer_thresh[0]:
                    self.generate_label(0, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)

                if length > layer_thresh[0] and length <= layer_thresh[1]:

                    self.generate_label(1, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)

                if length > layer_thresh[1]:
                    self.generate_label(2, self.gt_tensor, c_x_r, c_y_r, len_w, len_h, box_w, box_h, angle,
                                        ymin, xmax, ymax, xmin, c_x, c_y, a1, a2, a3, a4,
                                        gt_label, class_id)

        label_sbbox, label_mbbox, label_lbbox = self.gt_tensor
        return label_sbbox, label_mbbox, label_lbbox


if __name__ == '__main__':
    from torch.utils.data import DataLoader

    train_dataset = Construct_Dataset(anno_file_name="ssdd", img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
    train_dataloader = DataLoader(train_dataset, batch_size=1, num_workers=1, shuffle=False)
    for i, (imgs, label_sbbox, label_mbbox, label_lbbox) in enumerate(train_dataloader):
        continue


