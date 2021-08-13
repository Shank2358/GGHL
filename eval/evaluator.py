import shutil
import time
from tqdm import tqdm

from dataload.augmentations import *
from eval import voc_eval
from utils.utils_basic import *
from utils.visualize import *


current_milli_time = lambda: int(round(time.time() * 1000))

class Evaluator(object):
    def __init__(self, model, visiual=True):
        self.classes = cfg.DATA["CLASSES"]
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'prediction')#预测结果的保存路径
        self.val_data_path = cfg.DATA_PATH
        self.conf_thresh = cfg.TEST["CONF_THRESH"]
        self.nms_thresh = cfg.TEST["NMS_THRESH"]
        self.val_shape = cfg.TEST["TEST_IMG_SIZE"]
        self.__visiual = visiual
        self.__visual_imgs = cfg.TEST["NUM_VIS_IMG"]
        self.model = model
        self.device = next(model.parameters()).device
        self.inference_time = 0.
        self.showheatmap = cfg.SHOW_HEATMAP

    def APs_voc(self, multi_test=False, flip_test=False):
        img_inds_file = os.path.join(self.val_data_path,  'ImageSets', cfg.TEST["EVAL_NAME"] + '.txt')
        with open(img_inds_file, 'r') as f:
            lines = f.readlines()
            img_inds = [line.strip() for line in lines]

        rewritepath = os.path.join(self.pred_result_path, 'voc')
        if os.path.exists(rewritepath):
            shutil.rmtree(rewritepath)
        os.mkdir(rewritepath)
        rewritepath = os.path.join(self.pred_result_path, 'voc/txt_all')
        if os.path.exists(rewritepath):
            shutil.rmtree(rewritepath)
        os.mkdir(rewritepath)

        i=0
        for img_ind in tqdm(img_inds):
            i=i+1
            img_path = os.path.join(self.val_data_path, 'JPEGImages', img_ind+'.png')
            #目标： 直接改成读txt的文件名，每一行读取
            img = cv2.imread(img_path)
            bboxes_prd = self.get_bbox(img, multi_test, flip_test)

            if bboxes_prd.shape[0] != 0 and self.__visiual and i <= self.__visual_imgs:
                boxes = bboxes_prd[..., :4]
                class_inds = bboxes_prd[..., 5].astype(np.int32)
                scores = bboxes_prd[..., 4]
                visualize_boxes(image=img, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
                path = os.path.join(self.pred_result_path, "imgs/{}.png".format(i))
                cv2.imwrite(path, img)

                self.__visual_imgs += 1

            alltxt = os.path.join(self.pred_result_path, 'voc/txt_all/')
            f1 = open(alltxt + img_ind + ".txt", "w")
            for bbox in bboxes_prd:
                coor = np.array(bbox[:4], dtype=np.int32)
                score = bbox[4]
                class_ind = int(bbox[5])
                #################################################
                class_name = self.classes[class_ind]
                score = '%.4f' % score
                xmin, ymin, xmax, ymax = map(str, coor)
                s = ' '.join([img_ind, score, xmin, ymin, xmax, ymax]) + '\n'
                with open(os.path.join(self.pred_result_path, 'voc', 'comp4_det_test_' + class_name + '.txt'), 'a') as f2:
                    f2.write(s)
                f1.write("%s %s %s %s %s %s\n" % (class_name, score, str(xmin), str(ymin), str(xmax), str(ymax)))

                color = np.zeros(3)
                if int(class_ind) == 0:
                    # 25 black
                    color = (255, 128, 0)#000
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

                # color = (0, 255, 0)
                #cv2.polylines(img, [points], 1, color, 2)
                # print(points)
                cv2.rectangle(img, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, 2)
                # c1 左上角 c2 右下角

            store_path = os.path.join(cfg.PROJECT_PATH, 'data/results/', img_ind + '.png')  ########
            # print(store_path)
            cv2.imwrite(store_path, img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])  #################

            f1.close()
        self.inference_time = 1.0 * self.inference_time/ len(img_inds)
        return self.__calc_APs(), self.inference_time

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

        bboxes = nms(bboxes, self.conf_thresh, self.nms_thresh)

        return bboxes

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
        pred_bbox = p_d.squeeze().cpu().numpy()
        bboxes = self.__convert_pred(pred_bbox, test_shape, (org_h, org_w), valid_scale)

        if self.showheatmap and len(img):
            self.__show_heatmap(beta[2], org_img)
        return bboxes

    def __get_img_tensor(self, img, test_shape):
        img = Resize((test_shape, test_shape), correct_box=False)(img, None).transpose(2, 0, 1)
        return torch.from_numpy(img[np.newaxis, ...]).float()


    def __convert_pred(self, pred_bbox, test_input_size, org_img_shape, valid_scale):
        """
        预测框进行过滤，去除尺度不合理的框
        """
        pred_coor = xywh2xyxy(pred_bbox[:, :4])
        pred_conf = pred_bbox[:, 8]
        pred_prob = pred_bbox[:, 9:]
        # (1)
        # (xmin_org, xmax_org) = ((xmin, xmax) - dw) / resize_ratio
        # (ymin_org, ymax_org) = ((ymin, ymax) - dh) / resize_ratio
        # 需要注意的是，无论我们在训练的时候使用什么数据增强方式，都不影响此处的转换方式
        # 假设我们对输入测试图片使用了转换方式A，那么此处对bbox的转换方式就是方式A的逆向过程
        org_h, org_w = org_img_shape
        resize_ratio = min(1.0 * test_input_size / org_w, 1.0 * test_input_size / org_h)
        dw = (test_input_size - resize_ratio * org_w) / 2
        dh = (test_input_size - resize_ratio * org_h) / 2
        pred_coor[:, 0::2] = 1.0 * (pred_coor[:, 0::2] - dw) / resize_ratio
        pred_coor[:, 1::2] = 1.0 * (pred_coor[:, 1::2] - dh) / resize_ratio

        # (2)将预测的bbox中超出原图的部分裁掉
        pred_coor = np.concatenate([np.maximum(pred_coor[:, :2], [0, 0]),
                                    np.minimum(pred_coor[:, 2:], [org_w - 1, org_h - 1])], axis=-1)
        # (3)将无效bbox的coor置为0
        invalid_mask = np.logical_or((pred_coor[:, 0] > pred_coor[:, 2]), (pred_coor[:, 1] > pred_coor[:, 3]))
        pred_coor[invalid_mask] = 0

        # (4)去掉不在有效范围内的bbox
        bboxes_scale = np.sqrt(np.multiply.reduce(pred_coor[:, 2:4] - pred_coor[:, 0:2], axis=-1))
        scale_mask = np.logical_and((valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1]))

        # (5)将score低于score_threshold的bbox去掉

        classes = np.argmax(pred_prob, axis=-1)
        scores = pred_conf * pred_prob[np.arange(len(pred_coor)), classes]
        #print("aaaaaaaaaa", pred_coor.shape, scale_mask.shape, score_mask.shape)
        score_mask = scores > self.conf_thresh

        mask = np.logical_and(scale_mask, score_mask)
        coors = pred_coor[mask]
        scores = scores[mask]
        classes = classes[mask]

        bboxes = np.concatenate([coors, scores[:, np.newaxis], classes[:, np.newaxis]], axis=-1)

        return bboxes


    def __calc_APs(self, iou_thresh=0.5, use_07_metric=False):
        """
        计算每个类别的ap值
        :param iou_thresh:
        :param use_07_metric:
        :return:dict{cls:ap}
        """
        filename = os.path.join(self.pred_result_path, 'voc', 'comp4_det_test_{:s}.txt')
        cachedir = os.path.join(self.pred_result_path, 'voc', 'cache')
        annopath = os.path.join(self.val_data_path, 'Annotations/{:s}.xml')
        imagesetfile = os.path.join(self.val_data_path, 'ImageSets', cfg.TEST["EVAL_NAME"]+'.txt')
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

        return APs
