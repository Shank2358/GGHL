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
from multiprocessing.dummy import Pool as ThreadPool        # 线程池
from collections import defaultdict

current_milli_time = lambda: int(round(time.time() * 1000))

class Evaluator(object):
    def __init__(self, model, visiual=True):
        self.classes = cfg.DATA["CLASSES"]
        self.classes_num = cfg.DATA["NUM"]
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'predictionR')#预测结果的保存路径
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
        self.showheatmap = cfg.SHOW_HEATMAP
        self.iouthresh_test = cfg.TEST["IOU_THRESHOLD"]
        self.topk = 150
        self.use_nms = True

        self.multi_test = cfg.TEST["MULTI_SCALE_TEST"]
        self.flip_test = cfg.TEST["FLIP_TEST"]
        self.final_result = defaultdict(list)

    def APs_voc(self):
        filename = cfg.TEST["EVAL_NAME"]+'.txt'
        img_inds_file = os.path.join(self.val_data_path, 'ImageSets', filename)
        with open(img_inds_file, "r") as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines] # 读取文件名

        rewritepath = os.path.join(self.pred_result_path, 'voc')
        if os.path.exists(rewritepath):
            shutil.rmtree(rewritepath)
        os.mkdir(rewritepath)

        imgs_count = len(img_inds)
        cpu_nums = cfg.TEST["NUMBER_WORKERS"] #multiprocessing.cpu_count()
        pool = ThreadPool(cpu_nums)
        with tqdm(total=imgs_count) as pbar:
            for i, _ in enumerate(pool.imap_unordered(self.APs_voc_Single, img_inds)):
                pbar.update()
        for class_name in self.final_result:
            with open(os.path.join(self.pred_result_path, 'voc', class_name + '.txt'), 'a') as f:
                str_result = ''.join(self.final_result[class_name])
                f.write(str_result)
        self.inference_time = 1.0 * self.inference_time / len(img_inds)
        APs, r, p = self.__calc_APs(iou_thresh=self.iouthresh_test)
        return APs, r, p, self.inference_time
        #return self.__calc_APs(), self.inference_time

    def APs_voc_Single(self, img_ind):
        img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind + '.png')  # 路径+JPEG+文件名############png
        # 目标： 直接改成读txt的文件名，每一行读取
        #print(img_path)
        img = cv2.imread(img_path)
        bboxes_prd = self.get_bbox(img, self.multi_test, self.flip_test)
        '''
        if bboxes_prd.shape[0] != 0 and self.__visiual and self.__visual_imgs < 100:
            boxes = bboxes_prd[..., :4]
            boxes_rota = bboxes_prd[..., 4:8]

            scores = bboxes_prd[..., 8]

            class_inds = bboxes_prd[..., 9].astype(np.int32)

            visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
            path = os.path.join(cfg.PROJECT_PATH, "data/results/{}.jpg".format(self.__visual_imgs))
            cv2.imwrite(path, img)

            self.__visual_imgs += 1
        '''

        for bbox in bboxes_prd:
            coor = np.array(bbox[:4], dtype=np.int32)
            a_rota = np.array(bbox[4:8], dtype=np.float64)
            x1 = a_rota[0] * (coor[2]-coor[0]) + coor[0]
            y1 = coor[1]
            x2 = coor[2]
            y2 = a_rota[1] * (coor[3]-coor[1]) + coor[1]
            x3 = coor[2] - a_rota[2] * (coor[2]-coor[0])
            y3 = coor[3]
            x4 = coor[0]
            y4 = coor[3] - a_rota[3] * (coor[3]-coor[1])
            score = bbox[8]
            class_ind = int(bbox[9])
            class_name = self.classes[class_ind]
            score = '%.4f' % score
            s = ' '.join([img_ind, score, str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)),
                          str(int(x3)), str(int(y3)), str(int(x4)), str(int(y4))]) + '\n'
            self.final_result[class_name].append(s)
            color = np.zeros(3)
            points = np.array(
                [[int(x1), int(y1)], [int(x2), int(y2)], [int(x3), int(y3)], [int(x4), int(y4)]])
            if int(class_ind) == 0:
                # 25 black
                color = (0, 0, 0)
            elif int(class_ind) == 1:
                # 1359 blue
                color = (255, 0, 0)
            elif int(class_ind) == 2:
                # 639 Yellow
                color = (0, 255, 255)
            elif int(class_ind) == 3:
                # 4371 red
                color = (0, 0, 255)
            elif int(class_ind) == 4:
                # 3025 green
                color = (0, 255, 0)
            elif int(class_ind) == 5:
                # 1359 blue
                color = (255, 0, 0)
            elif int(class_ind) == 6:
                # 639 Yellow
                color = (0, 128, 255)
            elif int(class_ind) == 7:
                # 4371 red
                color = (0, 0, 128)
            elif int(class_ind) == 8:
                # 3025 green
                color = (0, 128, 0)
            elif int(class_ind) == 9:
                # 1359 blue
                color = (128, 0, 0)
            elif int(class_ind) == 10:
                # 639 Yellow
                color = (128, 128, 0)
            elif int(class_ind) == 11:
                # 4371 red
                color = (0, 128, 128)
            elif int(class_ind) == 12:
                # 3025 green
                color = (128, 128, 0)
            elif int(class_ind) == 13:
                # 1359 blue
                color = (0, 255, 128)
            elif int(class_ind) == 14:
                # 639 Yellow
                color = (255, 128, 255)
            #color = (0, 255, 0)
            cv2.polylines(img, [points], 1, color, 2)
            #print(points)
            #cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
            # c1 左上角 c2 右下角
        store_path = os.path.join(cfg.PROJECT_PATH, 'dataR/results/', img_ind + '.png')########
        cv2.imwrite(store_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])#################

    def get_bbox(self, img, multi_test=False, flip_test=False):
        if multi_test:
            test_input_sizes = range(cfg.TEST["MULTI_TEST_RANGE"][0], cfg.TEST["MULTI_TEST_RANGE"][1], cfg.TEST["MULTI_TEST_RANGE"][2])
            bboxes_list = []
            for test_input_size in test_input_sizes:
                valid_scale =(0, np.inf)
                bboxes_list.append(self.__predict(img, test_input_size, valid_scale))
                if flip_test:
                    bboxes_flip = self.__predict(img[:, ::-1], test_input_size, valid_scale)
                    bboxes_flip[:, [0, 2]] = img.shape[1] - bboxes_flip[:, [2, 0]]
                    bboxes_list.append(bboxes_flip)
            bboxes = np.row_stack(bboxes_list)
        else:
            bboxes = self.__predict(img, self.val_shape, (0, np.inf))

        #bboxes = nms_glid(bboxes, self.conf_thresh, self.nms_thresh)#################################

        return bboxes

    def _gather_feat(self, feat, ind, mask=None):
        dim = feat.size(2)
        ind = ind.unsqueeze(2).expand(ind.size(0), ind.size(1), dim)
        feat = feat.gather(1, ind)
        if mask is not None:
            mask = mask.unsqueeze(2).expand_as(feat)
            feat = feat[mask]
            feat = feat.view(-1, dim)
        return feat

    def _topk(self, scores):
        B, C, H, W = scores.size()
        topk_scores, topk_inds = torch.topk(scores.view(B, C, -1), self.topk)
        topk_inds = topk_inds % (H * W)
        topk_score, topk_ind = torch.topk(topk_scores.view(B, -1), self.topk)
        topk_clses = (topk_ind / self.topk).int()
        topk_inds = self._gather_feat(topk_inds.view(B, -1, 1), topk_ind).view(B, self.topk)
        return topk_score, topk_inds, topk_clses

    def nms(self, dets, scores):
        """"Pure Python NMS baseline."""
        x1 = dets[:, 0]  #xmin
        y1 = dets[:, 1]  #ymin
        x2 = dets[:, 2]  #xmax
        y2 = dets[:, 3]  #ymax
        areas = (x2 - x1) * (y2 - y1)                 # the size of bbox
        order = scores.argsort()[::-1]                        # sort bounding boxes by decreasing order
        keep = []                                             # store the final bounding boxes
        while order.size > 0:
            i = order[0]                                      #the index of the bbox with highest confidence
            keep.append(i)                                    #save it to keep
            xx1 = np.maximum(x1[i], x1[order[1:]])
            yy1 = np.maximum(y1[i], y1[order[1:]])
            xx2 = np.minimum(x2[i], x2[order[1:]])
            yy2 = np.minimum(y2[i], y2[order[1:]])
            w = np.maximum(1e-28, xx2 - xx1)
            h = np.maximum(1e-28, yy2 - yy1)
            inter = w * h
            # Cross Area / (bbox + particular area - Cross Area)
            ovr = inter / (areas[i] + areas[order[1:]] - inter + 1e-14)
            #reserve all the boundingbox whose ovr less than thresh
            inds = np.where(ovr <= self.nms_thresh)[0]
            order = order[inds + 1]
        return keep

    def __predict(self, img, test_shape, valid_scale):
        org_img = np.copy(img)
        org_h, org_w, _ = org_img.shape
        img = self.__get_img_tensor(img, test_shape).to(self.device)
        self.model.eval()
        with torch.no_grad():
            start_time = current_milli_time()
            if self.showheatmap: _, p_d, beta = self.model(img)
            else: _, p_d = self.model(img)
            self.inference_time += (current_milli_time() - start_time)
            pred_bbox_list = []
            shape = img.shape[2]
            for i in range(3):
                stride = int(self.strides[i])
                pred_shape = int(shape/stride)#int(self.val_shape/stride)
                p_d_temp = p_d[i]
                cls_pred = p_d_temp[:, 14:].view(1, pred_shape, pred_shape, self.classes_num).permute(0, 3, 1, 2)
                conf_pred = p_d_temp[:, 13:14].view(1, pred_shape, pred_shape, 1).permute(0, 3, 1, 2)
                '''
                plt.figure("Image")  # 图像窗口名称
                plt.imshow(org_img)
                img1 = p_d_temp[:, 9:].view(1, pred_shape, pred_shape, self.classes_num).detach().cpu()[0, :, :, 5:6]
                plt.figure("Image1")  # 图像窗口名称
                plt.imshow(img1)
                pred_result = cls_pred.permute(0, 2, 3, 1).detach().cpu()[0, :, :, 5:6]
                plt.figure("Image2")  # 图像窗口名称
                plt.imshow(pred_result)
                plt.show()'''

                # simple nms
                pred_scores = conf_pred*cls_pred
                #hmax = F.max_pool2d(pred_scores, kernel_size=1, padding=0, stride=1)#hmax = F.max_pool2d(cls_pred, kernel_size=5, padding=2, stride=1)
                #keep = (hmax == pred_scores).float()
                #pred_scores *= keep
                '''
                plt.figure("Image")  # 图像窗口名称
                plt.imshow(org_img)
                img1 = p_d_temp[:, 9:].view(1, pred_shape, pred_shape, self.classes_num).detach().cpu()[0, :, :, 5:6]
                plt.figure("Image1")  # 图像窗口名称
                plt.imshow(img1)
                pred_result = cls_pred.permute(0, 2, 3, 1).detach().cpu()[0, :, :, 5:6]
                plt.figure("Image2")  # 图像窗口名称
                plt.imshow(pred_result)
                plt.show()'''

                # topk
                topk_scores, topk_inds, topk_clses = self._topk(pred_scores)
                bbox_pred = p_d_temp[:, :].view(1, -1, 9 + 4 + self.classes_num + 1)[0]#.contiguous()
                topk_scores = topk_scores[0].cpu().numpy()
                topk_ind = topk_clses[0].cpu().numpy()
                topk_bbox_pred = bbox_pred[topk_inds[0]].cpu().numpy()
                if self.use_nms:
                    # nms
                    keep = np.zeros(len(topk_bbox_pred), dtype=np.int)
                    for i in range(15):
                        inds = np.where(topk_ind == i)[0]
                        if len(inds) == 0:
                            continue
                        c_bboxes = topk_bbox_pred[inds]
                        c_scores = topk_scores[inds]
                        c_keep = self.nms(c_bboxes, c_scores)
                        keep[inds[c_keep]] = 1
                    keep = np.where(keep > 0)
                    topk_bbox_pred = topk_bbox_pred[keep]
                    topk_scores = topk_scores[keep]
                    topk_ind = topk_ind[keep]
                topk_scores = topk_scores[:,np.newaxis]
                topk_ind = topk_ind[:,np.newaxis]

                pred_bbox = np.concatenate((topk_bbox_pred, topk_scores, topk_ind), axis=-1)
                pred_bbox_list.append(pred_bbox)

        pred_bbox_set = np.concatenate((pred_bbox_list[0], pred_bbox_list[1], pred_bbox_list[2]), axis=0)

        bboxes = self.__convert_pred(pred_bbox_set, test_shape, (org_h, org_w), valid_scale)
        return bboxes

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()

    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        #label: xywh(没有scale, 0 - 4), a1 - a4(4 - 8), r(8), conf(9), one_hot_smooth(10...)
        pred_coor = xywh2xyxy(pred_bbox[:, :4]) #xywh2xyxy
        pred_score = pred_bbox[:, 14 + self.classes_num:]

        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio
        pred_rotaxy = pred_bbox[:, 4:8]
        pred_r = pred_bbox[:, 8:9]
        zero = np.zeros_like(pred_rotaxy)
        pred_rotaxy = np.where(pred_r > 0.9, zero, pred_rotaxy) #0.8

        # (2)将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]), np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0
        pred_rotaxy[invalid_mask] = 0############################
        # (4)去掉不在有效范围内的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))
        # (5)将score低于score_threshold的bbox去掉
        classes = pred_score[:,1:2]#np.argmax(pred_prob, axis=-1)
        scores = pred_score[:,0:1] #np.max(pred_prob, axis=-1) #* pred_prob[np.arange(len(pred_coor)), classes]
        score_mask = scores > self.conf_thresh
        mask = np.logical_and(scale_mask, score_mask.squeeze(-1))
        coors = pred_coor[mask]
        coors_rota = pred_rotaxy[mask]
        scores = scores[mask]
        classes = classes[mask]
        #pred_prob = pred_prob[mask]
        bboxes = np.concatenate([coors, coors_rota, scores, classes], axis=-1)#######################
        return bboxes

    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        计算每个类别的ap值
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        filename = os.path.join(self.pred_result_path, 'voc/{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'voc', 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations/{:s}.txt')
        imagesetfile = os.path.join(self.val_data_path, 'ImageSets', cfg.TEST["EVAL_NAME"]+'.txt')
        #print(annopath)
        APs = {}
        Recalls = {}
        Precisions = {}
        for i, cls in enumerate(self.classes):
            R, P, AP = voc_eval.voc_eval(filename, annopath, imagesetfile, cls, cachedir, iou_thresh, use_07_metric)#调用voc_eval.py的函数进行计算
            APs[cls] = AP
            Recalls[cls] = R
            Precisions[cls] = P
        if os.path.exists(cachedir):
            shutil.rmtree(cachedir)
        return APs, Recalls, Precisions
