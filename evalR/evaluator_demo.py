import shutil
import time
from tqdm import tqdm
import torch.nn.functional as F
from dataloadR.augmentations import *
from evalR import voc_eval
from utils.utils_basic import *
from utils.visualize import *
import time
import multiprocessing
from multiprocessing.dummy import Pool as ThreadPool  # 线程池
from collections import defaultdict

current_milli_time = lambda: int(round(time.time() * 1000))


class Evaluator(object):
    def __init__(self, model, visiual=True):
        self.classes = cfg.DATA["CLASSES"]
        self.classes_num = cfg.DATA["NUM"]
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'predictionR')  # 预测结果的保存路径
        self.val_data_path = cfg.DATA_PATH
        self.strides = cfg.MODEL["STRIDES"]
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.val_shape = cfg.TEST["TEST_IMG_SIZE"]
        self.__visiual = visiual
        self.__visual_imgs = cfg.TEST["NUM_VIS_IMG"]
        self.model = model
        self.device = next(model.parameters()).device
        self.inference_time = 0.
        self.iouthresh_test = cfg.TEST["IOU_THRESHOLD"]
        self.multi_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.flip_test = cfg.TEST["FLIP_TEST"]

    def Test_single_img(self, img_id):
        img_path = os.path.join(self.val_data_path, 'JPEGImages', img_id + '.png')  ###测试图像路径
        img = cv2.imread(img_path)
        bboxes_prd = self.get_bbox(img, self.multi_test, self.flip_test)
        for bbox in bboxes_prd:
            x1 = bbox[0]
            y1 = bbox[1]
            x2 = bbox[2]
            y2 = bbox[3]
            x3 = bbox[4]
            y3 = bbox[5]
            x4 = bbox[6]
            y4 = bbox[7]
            score = bbox[8]
            class_ind = int(bbox[9])
            class_name = self.classes[class_ind]
            score = '%.4f' % score
            ''''''
            color = np.zeros(3)
            points = np.array(
                [[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]])
            if int(class_ind) == 0: color = (64, 0, 0)
            elif int(class_ind) == 1: color = (255, 0, 0)
            elif int(class_ind) == 2: color = (0, 255, 255)
            elif int(class_ind) == 3: color = (0, 0, 255)
            elif int(class_ind) == 4: color = (0, 255, 0)
            elif int(class_ind) == 5: color = (255, 0, 0)
            elif int(class_ind) == 6: color = (0, 128, 255)
            elif int(class_ind) == 7: color = (0, 0, 128)
            elif int(class_ind) == 8: color = (0, 128, 0)
            elif int(class_ind) == 9: color = (128, 0, 0)
            elif int(class_ind) == 10: color = (128, 128, 0)
            elif int(class_ind) == 11:  color = (0, 128, 128)
            elif int(class_ind) == 12: color = (128, 128, 0)
            elif int(class_ind) == 13: color = (0, 255, 128)
            elif int(class_ind) == 14: color = (255, 128, 255)
            cv2.polylines(img, [points], 1, color, 2)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, class_name + ' ' + score[:4], (int(float(x1)), int(float(y1))), font, 0.3, (255, 255, 255), 1)
        store_path = os.path.join(self.pred_result_path, 'imgs', img_id + '.png')  #保存结果路径
        cv2.imwrite(store_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(cfg.TEST["MULTI_TEST_RANGE"][0], cfg.TEST["MULTI_TEST_RANGE"][1],
                                     cfg.TEST["MULTI_TEST_RANGE"][2])
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale = (0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))
        bboxes = self.non_max_suppression_4points(bboxes, self.conf_thresh, self.nms_thresh, multi_label=False)
        return bboxes[0].cpu().numpy()

    def __predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape
        img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            start_time = current_milli_time()
            _, p_d = self.model(img)
            self.inference_time += (current_milli_time() - start_time)

        pred_bbox = p_d.squeeze()
        bboxes = self.convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)
        return bboxes

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()

    def convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        pred_xyxy = xywh2xyxy(pred_bbox[:, :4])  # xywh2xyxy
        pred_conf = pred_bbox[:, 13]
        pred_prob = pred_bbox[:, 14:]
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_xyxy[:, 0::2] = 1.0 * (pred_xyxy[:, 0::2] - dw) / resize_ratio
        pred_xyxy[:, 1::2] = 1.0 * (pred_xyxy[:, 1::2] - dh) / resize_ratio
        pred_s = pred_bbox[:, 4:8]
        pred_r = pred_bbox[:, 8:9]
        zero = torch.zeros_like(pred_s)
        pred_s = torch.where(pred_r > 0.9, zero, pred_s)
        # (2)将预测的bbox中超出原图的部分裁掉
        device = pred_bbox.device
        pred_xyxy = torch.cat(
            [torch.maximum(pred_xyxy[:, :2], torch.tensor([0, 0]).to(device)),
             torch.minimum(pred_xyxy[:, 2:], torch.tensor([org_w - 1, org_h - 1]).to(device))], dim=-1)

        invalid_mask = torch.logical_or((pred_xyxy[:, 0] > pred_xyxy[:, 2]), (pred_xyxy[:, 1] > pred_xyxy[:, 3]))
        pred_xyxy[invalid_mask] = 0
        pred_s[invalid_mask] = 0
        # (4)去掉不在有效范围内的bbox
        bboxes_scale = torch.sqrt(
            (pred_xyxy[..., 2:3] - pred_xyxy[..., 0:1]) * (pred_xyxy[..., 3:4] - pred_xyxy[..., 1:2]))
        scale_mask = torch.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1])).squeeze(-1)

        # (5)将score低于score_threshold的bbox去掉
        classes = torch.argmax(pred_prob, dim=-1)
        scores = pred_conf * pred_prob[torch.arange(len(pred_xyxy)), classes]
        score_mask = scores > self.conf_thresh
        mask = torch.logical_and(scale_mask, score_mask)

        pred_xyxy = pred_xyxy[mask]
        pred_s = pred_s[mask]

        pred_conf = pred_conf[mask]

        # classes = classes[mask]
        pred_prob = pred_prob[mask]

        x1 = pred_s[:, 0:1] * (pred_xyxy[:, 2:3] - pred_xyxy[:, 0:1]) + pred_xyxy[:, 0:1]
        y1 = pred_xyxy[:, 1:2]
        x2 = pred_xyxy[:, 2:3]
        y2 = pred_s[:, 1:2] * (pred_xyxy[:, 3:4] - pred_xyxy[:, 1:2]) + pred_xyxy[:, 1:2]
        x3 = pred_xyxy[:, 2:3] - pred_s[:, 2:3] * (pred_xyxy[:, 2:3] - pred_xyxy[:, 0:1])
        y3 = pred_xyxy[:, 3:4]
        x4 = pred_xyxy[:, 0:1]
        y4 = pred_xyxy[:, 3:4] - pred_s[:, 3:4] * (pred_xyxy[:, 3:4] - pred_xyxy[:, 1:2])
        coor4points = torch.cat([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)

        bboxes = torch.cat([coor4points, pred_conf.unsqueeze(-1), pred_prob], dim=-1)

        # print(bboxes)
        bs = 1  # cfg.TEST["NUMBER_WORKERS"]
        # _, out_num = bboxes.shape[1]
        # print(bboxes.shape)
        bboxes = bboxes.view(bs, -1, bboxes.shape[1])
        return bboxes

    def non_max_suppression_4points(self,
                                    prediction, conf_thres=0.2, iou_thres=0.45, merge=False, classes=None,
                                    multi_label=True, agnostic=False,
                                    without_iouthres=False
                                    ):
        """
        Performs Rotate-Non-Maximum Suppression (RNMS) on inference results；
        @param prediction: size=(batch_size, num, [xywh,score,num_classes,num_angles])
        @param conf_thres: 置信度阈值
        @param iou_thres:  IoU阈值
        @param merge: None
        @param classes: None
        @param agnostic: 进行nms是否将所有类别框一视同仁，默认False
        @param without_iouthres : 本次nms不做iou_thres的标志位  默认为False
        @return:
                output: 经nms后的旋转框(batch_size, num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]
        """
        # prediction :(batch_size, num_boxes, [xywh,score,num_classes,num_angles])
        nc = prediction[0].shape[1] - 9
        # class_index = nc + 9
        # xc : (batch_size, num_boxes) 对应位置为1说明该box超过置信度
        xc = prediction[..., 8] > conf_thres  # candidates

        # Settings
        min_wh, max_wh = 2, 4096  # (pixels) minimum and maximum box width and height
        max_det = 500  # maximum number of detections per image
        time_limit = 10.0  # seconds to quit after
        redundant = True  # require redundant detections 要求冗余检测
        multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

        t = time.time()
        # output: (batch_size, ?)
        output = [torch.zeros((0, 10), device=prediction.device)] * prediction.shape[0]
        for xi, x in enumerate(prediction):  # image index, image inference
            # Apply constraints
            # x ： (num_boxes, [xywh, score, num_classes, num_angles])
            # x[((x[..., 2:4] < min_wh) | (x[..., 2:4] > max_wh)).any(1), 4] = 0  # width-height
            x = x[xc[xi]]  # 取出数组中索引为True的的值即将置信度符合条件的boxes存入x中   x -> (num_confthres_boxes, no)
            # If none remain process next image
            if not x.shape[0]:
                continue
            # Compute conf

            x[:, 9:] = x[:, 9:] * x[:, 8:9]  # # conf = obj_conf * cls_conf
            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            # angle = x[:, class_index:]  # angle.size=(num_confthres_boxes, [num_angles])
            # torch.max(angle,1) 返回每一行中最大值的那个元素，且返回其索引（返回最大元素在这一行的列索引）
            # angle_value, angle_index = torch.max(angle, 1, keepdim=True)  # size都为 (num_confthres_boxes, 1)
            # box.size = (num_confthres_boxes, [xywhθ])  θ∈[0,179]
            # box = torch.cat((x[:, :4], angle_index), 1)
            box = x[:, :8]
            if multi_label:
                # nonzero ： 取出每个轴的索引,默认是非0元素的索引（取出括号公式中的为True的元素对应的索引） 将索引号放入i和j中
                # i：num_boxes该维度中的索引号，表示该索引的box其中有class的conf满足要求  length=x中满足条件的所有坐标数量
                # j：num_classes该维度中的索引号，表示某个box中是第j+1类物体的conf满足要求  length=x中满足条件的所有坐标数量
                i, j = (x[:, 9:] > conf_thres).nonzero(as_tuple=False).T
                # 按列拼接  list x：(num_confthres_boxes, [xywhθ]+[conf]+[classid]) θ∈[0,179]
                x = torch.cat((box[i], x[i, j + 9, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 9:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[conf.view(-1) > conf_thres]
            if without_iouthres:  # 不做nms_iou
                output[xi] = x
                continue
            # Filter by class 按类别筛选
            if classes:
                x = x[(x[:, 9:] == torch.tensor(classes, device=x.device)).any(1)]  # any（1）函数表示每行满足条件的返回布尔值
            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue
            # Sort by confidence
            c = x[:, 9:] * (0 if agnostic else max_wh)  # classes
            boxes_4points, scores = x[:, :8] + c, x[:, 8]
            i = np.array(py_cpu_nms_poly_fast(np.double(boxes_4points.cpu().numpy()), scores.cpu().numpy(), iou_thres))
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            temp = x[i].clone()
            # output[xi] = x[i]  # 根据nms索引提取x中的框  x.size=(num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]
            output[xi] = temp
            if (time.time() - t) > time_limit:
                break  # time limit exceeded
        return output
