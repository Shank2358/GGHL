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
        self.final_result = defaultdict(list)
        self.alpha = 0.4

    def APs_voc(self):
        filename = cfg.TEST["EVAL_NAME"] + '.txt'
        img_inds_file = os.path.join(self.val_data_path, 'ImageSets', filename)
        with open(img_inds_file, "r") as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]  # 读取文件名

        rewritepath = os.path.join(self.pred_result_path, 'voc')
        if os.path.exists(rewritepath):
            shutil.rmtree(rewritepath)
        os.mkdir(rewritepath)

        imgs_count = len(img_inds)
        cpu_nums = cfg.TEST["NUMBER_WORKERS"]  # multiprocessing.cpu_count()
        pool = ThreadPool(cpu_nums)
        torch.backends.cudnn.enabled = False
        with tqdm(total=imgs_count) as pbar:
            for i, _ in enumerate(pool.imap_unordered(self.APs_voc_Single, img_inds)):
                pbar.update()
        for class_name in self.final_result:
            with open(os.path.join(self.pred_result_path, 'voc', class_name + '.txt'), 'a') as f:
                str_result = ''.join(self.final_result[class_name])
                f.write(str_result)
        APs, r, p = self.__calc_APs(iou_thresh=self.iouthresh_test)
        #return APs, r, p, self.inference_time
        return APs, self.inference_time

    def APs_voc_Single(self, img_ind):
        img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind + '.png')  # 路径+JPEG+文件名############png
        #print(img_path+'\n')
        img = cv2.imread(img_path)
        #print('\n')
        bboxes_prd = self.get_bbox(img, self.multi_test, self.flip_test)
        for bbox in bboxes_prd:
            x1 = bbox[0]#]a_rota[0] * (coor[2] - coor[0]) + coor[0]
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
            #print(class_name, score, str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(x3)), str(int(y3)), str(int(x4)), str(int(y4)))
            score = '%.4f' % score
            s = ' '.join([img_ind, score, str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)),
                          str(int(x3)), str(int(y3)), str(int(x4)), str(int(y4))]) + '\n'
            self.final_result[class_name].append(s)
            ''''''
            color = np.zeros(3)
            points = np.array(
                [[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]])

            # if int(class_ind) == 0: color = (255, 128, 0)
            # elif int(class_ind) == 1: color = (255, 0, 0)
            # elif int(class_ind) == 2: color = (0, 255, 255)
            # elif int(class_ind) == 3: color = (0, 0, 255)
            # elif int(class_ind) == 4: color = (0, 255, 0)
            # elif int(class_ind) == 5: color = (255, 0, 0)
            # elif int(class_ind) == 6: color = (0, 128, 255)
            # elif int(class_ind) == 7: color = (0, 0, 128)
            # elif int(class_ind) == 8: color = (0, 128, 0)
            # elif int(class_ind) == 9: color = (128, 0, 0)
            # elif int(class_ind) == 10: color = (128, 128, 0)
            # elif int(class_ind) == 11:  color = (0, 128, 128)
            # elif int(class_ind) == 12: color = (128, 128, 0)
            # elif int(class_ind) == 13: color = (0, 255, 128)
            # elif int(class_ind) == 14:  color = (255, 128, 255)
            # elif int(class_ind) == 15: color = (64, 0, 128)
            # elif int(class_ind) == 16: color = (64, 128, 0)
            # elif int(class_ind) == 17: color = (128, 64, 128)
            # elif int(class_ind) == 18: color = (255, 128, 64)
            # elif int(class_ind) == 19:  color = (64, 128, 128)

            if int(class_ind) == 0: color = (103, 87, 239)#	255,64,64
            elif int(class_ind) == 1: color = (112, 125, 241)
            elif int(class_ind) == 2: color = (130, 171, 255)
            elif int(class_ind) == 3: color = (95, 238, 209)
            elif int(class_ind) == 4: color = (165, 0, 255)

            elif int(class_ind) == 5: color = (132,68, 255)
            elif int(class_ind) == 6: color = (153,123, 254)
            elif int(class_ind) == 7: color = (207,69, 254)
            elif int(class_ind) == 8: color = (215,81, 243)
            elif int(class_ind) == 9: color = (241,184, 241)

            elif int(class_ind) == 10: color = (90,205, 106)
            elif int(class_ind) == 11:  color = (149,237, 100)
            elif int(class_ind) == 12: color = (191,255, 0)
            elif int(class_ind) == 13: color = (238,238, 175)
            elif int(class_ind) == 14:  color = (241,237, 184)

            elif int(class_ind) == 15: color = (179, 113, 60)
            elif int(class_ind) == 16: color = (178, 170, 32)
            elif int(class_ind) == 17: color = (241, 204, 184)
            elif int(class_ind) == 18: color = (255, 149, 221)
            elif int(class_ind) == 19: color = (211, 143, 184)
            cv2.polylines(img, [points], 1, color, 4)
            font = cv2.FONT_HERSHEY_SIMPLEX
            img = cv2.putText(img, class_name + ' ' + score[:4], (int(float(x1)), int(float(y1))), font, 0.3, (255, 255, 255), 1)
        store_path = os.path.join(self.pred_result_path, 'imgs', img_ind + '.jpg')
        cv2.imwrite(store_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])

    def get_bbox(self, img, multi_test=False, flip_test=False):
        # start_time = current_milli_time()
        if multi_test:
            test_input_sizes = range(cfg.TEST["MULTI_TEST_RANGE"][0], cfg.TEST["MULTI_TEST_RANGE"][1], cfg.TEST["MULTI_TEST_RANGE"][2])
            #bboxes_list = []
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))
            for test_input_size in test_input_sizes:
                if test_input_size != self.val_shape:
                #print(test_input_size)
                #valid_scale = (0, np.inf)
                    bboxes1 = self.__predict(img, test_input_size, (0, np.inf))
                    bboxes = torch.cat([bboxes, bboxes1], dim=1)
                #bboxes1 = []
                #print(bboxes.shape)
                #bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                # if flip_test:
                #     bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                #     bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                #     bboxes_list.append(bboxes_flip)
            #print(bboxes_list)
            #bboxes = torch.stack(bboxes, dim=1)
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
            #_, _, _, p_d, _ = self.model(img)
            self.inference_time += (current_milli_time() - start_time)

        pred_bbox = p_d.squeeze()#.cpu().numpy()
        bboxes = self.convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)
        # bboxes = self.__convert_pred(pred_bbox_set, test_shape, (org_h, org_w), valid_scale)
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
        pred_s= torch.where(pred_r > 0.9, zero, pred_s)
        # (2)将预测的bbox中超出原图的部分裁掉
        device = pred_bbox.device
        pred_xyxy = torch.cat(
            [torch.maximum(pred_xyxy[:, :2], torch.tensor([0, 0]).to(device)),
            torch.minimum(pred_xyxy[:, 2:], torch.tensor([org_w - 1, org_h - 1]).to(device))], dim=-1)

        invalid_mask = torch.logical_or((pred_xyxy[:, 0] > pred_xyxy[:, 2]), (pred_xyxy[:, 1] > pred_xyxy[:, 3]))
        pred_xyxy[invalid_mask] = 0
        pred_s[invalid_mask] = 0
        # (4)去掉不在有效范围内的bbox
        bboxes_scale = torch.sqrt((pred_xyxy[..., 2:3] - pred_xyxy[..., 0:1]) * (pred_xyxy[..., 3:4] - pred_xyxy[..., 1:2]))
        scale_mask = torch.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1])).squeeze(-1)

        # (5)将score低于score_threshold的bbox去掉
        classes = torch.argmax(pred_prob, dim=-1)
        scores = pred_conf * pred_prob[torch.arange(len(pred_xyxy)), classes]
        score_mask = scores > self.conf_thresh
        mask = torch.logical_and(scale_mask, score_mask)

        pred_xyxy = pred_xyxy[mask]
        pred_s = pred_s[mask]

        pred_conf = pred_conf[mask]

        #classes = classes[mask]
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

        #print(bboxes)
        bs = 1#cfg.TEST["NUMBER_WORKERS"]
        #_, out_num = bboxes.shape[1]
        #print(bboxes.shape)
        bboxes = bboxes.view(bs, -1, bboxes.shape[1])
        return bboxes

    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        filename = os.path.join(self.pred_result_path, 'voc/{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'voc', 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations/{:s}.txt')
        imagesetfile = os.path.join(self.val_data_path, 'ImageSets', cfg.TEST["EVAL_NAME"] + '.txt')
        # print(annopath)
        APs = {}
        Recalls = {}
        Precisions = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh,
                                         use_07_metric)  # 调用voc_eval.py的函数进行计算
            APs[cls] = AP
            Recalls[cls] = R
            Precisions[cls] = P
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)
        return APs, Recalls, Precisions

    def non_max_suppression_4points_old(self,
            prediction, conf_thres=0.2, iou_thres=0.45, merge=False, classes=None, multi_label=True, agnostic=False,
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

            x[:, 9:] = x[:, 9:] * x[:, 8:9] #  # conf = obj_conf * cls_conf
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

    def non_max_suppression_4points(self,
            prediction, conf_thres=0.2, iou_thres=0.45, merge=False, classes=None, multi_label=True, agnostic=False,
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
            x[:, 9:] = torch.sqrt(x[:, 9:]* x[:, 8:9])  # conf = obj_conf * cls_conf
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

            # TODO change this poly iou
            # poly 速度非常慢
            # i = np.array(py_cpu_nms_poly_fast(np.double(boxes_4points.cpu().numpy()), scores.cpu().numpy(), iou_thres))

            # TODO 八点转角度
            # 先转numpy
            boxes_4points_np = boxes_4points.clone().cpu().numpy()
            scores_for_cv2_nms = scores.cpu().numpy()
            boxes_for_cv2_nms = []
            for box_inds, _ in enumerate(boxes_4points_np):
                points = np.reshape(boxes_4points_np[box_inds, :], (4, 2))
                boxes_xy, boxes_wh, boxes_angle = cv2.minAreaRect(points)
                boxes_for_cv2_nms.append((boxes_xy, boxes_wh, boxes_angle))

            i = cv2.dnn.NMSBoxesRotated(boxes_for_cv2_nms, scores_for_cv2_nms, conf_thres, iou_thres)
            i = torch.from_numpy(i).type(torch.LongTensor)
            i = i.squeeze(axis=-1)

            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]

            temp = x[i].clone()
            # output[xi] = x[i]  # 根据nms索引提取x中的框  x.size=(num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]
            output[xi] = temp
            if (time.time() - t) > time_limit:
                break  # time limit exceeded
        return output
