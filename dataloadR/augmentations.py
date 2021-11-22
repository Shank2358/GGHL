# coding=utf-8
import cv2
import random
import numpy as np
import imgaug.augmenters as iaa


class HSV(object):
    def __init__(self, hgain=0.3, sgain=0.5, vgain=0.5, p=0.5):
        self.hgain = hgain
        self.sgain = sgain
        self.vgain = vgain
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            x = np.random.uniform(-1, 1, 3) * [self.hgain, self.sgain, self.vgain] + 1
            img_hsv = (cv2.cvtColor(img, cv2.COLOR_BGR2HSV) * x).clip(None, 255).astype(np.uint8)
            np.clip(img_hsv[:, :, 0], None, 179, out=img_hsv[:, :, 0])
            img = cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)
        return img, bboxes

class Blur(object):
    def __init__(self, sigma=1.3, p=0.15):
        self.sigma = sigma
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            blur_aug = iaa.GaussianBlur(sigma=(0, self.sigma))
            img = blur_aug.augment_image(img)
        return img, bboxes


class Grayscale(object):
    def __init__(self, grayscale=0.3, p=0.5):
        self.alpha = random.uniform(grayscale, 1.0)
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            gray_aug = iaa.Grayscale(alpha=(self.alpha, 1.0))
            img = gray_aug.augment_image(img)
        return img, bboxes


class Gamma(object):
    def __init__(self, intensity=0.2, p=0.3):
        self.intensity = intensity
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            gm = random.uniform(1 - self.intensity, 1 + self.intensity)
            img = np.uint8(np.power(img / float(np.max(img)), gm) * np.max(img))
        return img, bboxes

class Noise(object):
    def __init__(self, intensity=0.01, p=0.15):
        self.intensity = intensity
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            noise_aug = iaa.AdditiveGaussianNoise(scale=(0, self.intensity * 255))
            img = noise_aug.augment_image(img)
        return img, bboxes

class Sharpen(object):
    def __init__(self, intensity=0.15, p=0.2):
        self.intensity = intensity
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            sharpen_aug = iaa.Sharpen(alpha=(0.0, 1.0), lightness=(1 - self.intensity, 1 + self.intensity))
            img = sharpen_aug.augment_image(img)
        return img, bboxes

class Contrast(object):
    def __init__(self, intensity=0.15, p=0.3):
        self.intensity = intensity
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            contrast_aug = iaa.contrast.LinearContrast((1 - self.intensity, 1 + self.intensity))
            img = contrast_aug.augment_image(img)
        return img, bboxes

class RandomVerticalFilp(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, _, _ = img.shape
            img = img[::-1, :, :]
            bboxes[:, [1, 3]] = h_img - bboxes[:, [3, 1]]
            bboxes[:, [6, 8, 10, 12]] = h_img - bboxes[:, [6, 8, 10, 12]]
            bboxes[:, [5, 6, 9, 10]] = bboxes[:, [9, 10, 5, 6]]
            bboxes[:, [-1]] = 180 - bboxes[:, [-1]]
        return img, bboxes

class RandomHorizontalFilp(object):
    def __init__(self, p=0.5):
        self.p = p
    def __call__(self, img, bboxes):
        if random.random() < self.p:
            _, w_img, _ = img.shape
            img = img[:, ::-1, :]
            bboxes[:, [0, 2]] = w_img - bboxes[:, [2, 0]]
            bboxes[:, [5, 7, 9, 11]] = w_img - bboxes[:, [5, 7, 9, 11]]
            bboxes[:, [7, 8, 11, 12]] = bboxes[:, [11, 12, 7, 8]]
            bboxes[:, [-1]] = 180 - bboxes[:, [-1]]
        return img, bboxes

class RandomCrop(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]
            crop_xmin = max(0, int(max_bbox[0] - random.uniform(0, max_l_trans)))
            crop_ymin = max(0, int(max_bbox[1] - random.uniform(0, max_u_trans)))
            crop_xmax = min(w_img, int(max_bbox[2] + random.uniform(0, max_r_trans)))#
            crop_ymax = min(h_img, int(max_bbox[3] + random.uniform(0, max_d_trans)))#
            img = img[crop_ymin : crop_ymax, crop_xmin : crop_xmax]
            bboxes[:, [0, 2, 5, 7, 9, 11]] = bboxes[:, [0, 2, 5, 7, 9, 11]] - crop_xmin
            bboxes[:, [1, 3, 6, 8, 10, 12]] = bboxes[:, [1, 3, 6, 8, 10, 12]] - crop_ymin
        return img, bboxes

class RandomAffine(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img, bboxes):
        if random.random() < self.p:
            h_img, w_img, _ = img.shape
            max_bbox = np.concatenate([np.min(bboxes[:, 0:2], axis=0), np.max(bboxes[:, 2:4], axis=0)], axis=-1)
            max_l_trans = max_bbox[0]
            max_u_trans = max_bbox[1]
            max_r_trans = w_img - max_bbox[2]
            max_d_trans = h_img - max_bbox[3]

            tx = random.uniform(-(max_l_trans - 1), (max_r_trans - 1))
            ty = random.uniform(-(max_u_trans - 1), (max_d_trans - 1))

            M = np.array([[1, 0, tx], [0, 1, ty]])
            img = cv2.warpAffine(img, M, (w_img, h_img))

            bboxes[:, [0, 2, 5, 7, 9, 11]] = bboxes[:, [0, 2, 5, 7, 9, 11]] + tx
            bboxes[:, [1, 3, 6, 8, 10, 12]] = bboxes[:, [1, 3, 6, 8, 10, 12]] + ty
        return img, bboxes


class Resize(object):

    def __init__(self, target_shape, correct_box=True):
        self.h_target, self.w_target = target_shape
        self.correct_box = correct_box

    def __call__(self, img, bboxes):
        h_org , w_org , _= img.shape
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)

        resize_ratio = min(1.0 * self.w_target / w_org, 1.0 * self.h_target / h_org)
        resize_w = int(resize_ratio * w_org)
        resize_h = int(resize_ratio * h_org)
        image_resized = cv2.resize(img, (resize_w, resize_h))

        image_paded = np.full((self.h_target, self.w_target, 3), 128.0)
        dw = int((self.w_target - resize_w) / 2)
        dh = int((self.h_target - resize_h) / 2)
        image_paded[dh:resize_h + dh, dw:resize_w + dw, :] = image_resized
        image = image_paded / 255.0

        if self.correct_box:
            ################################x1-y4 trans
            bboxes[:, [0, 2, 5, 7, 9, 11]] = bboxes[:, [0, 2, 5, 7, 9, 11]] * resize_ratio + dw
            bboxes[:, [1, 3, 6, 8, 10, 12]] = bboxes[:, [1, 3, 6, 8, 10, 12]] * resize_ratio + dh
            return image, bboxes
        return image

class Mixup(object):
    def __init__(self, p=0.3):
        self.p = p

    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):

        if random.random() > self.p:
            lam = np.random.beta(1.5, 1.5)
            img = lam * img_org + (1 - lam) * img_mix
            bboxes_org = np.concatenate(
                [bboxes_org, np.full((len(bboxes_org), 1), lam)], axis=1)
            bboxes_mix = np.concatenate(
                [bboxes_mix, np.full((len(bboxes_mix), 1), 1 - lam)], axis=1)
            bboxes = np.concatenate([bboxes_org, bboxes_mix])

        else:
            img = img_org
            bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)

        return img, bboxes

class Mixup_no(object):
    def __call__(self, img_org, bboxes_org, img_mix, bboxes_mix):
        img = img_org
        bboxes = np.concatenate([bboxes_org, np.full((len(bboxes_org), 1), 1.0)], axis=1)
        return img, bboxes

class LabelSmooth(object):
    def __init__(self, delta=0.01):
        self.delta = delta

    def __call__(self, onehot, num_classes):
        return onehot * (1 - self.delta) + self.delta * 1.0 / num_classes
