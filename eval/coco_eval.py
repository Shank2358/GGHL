import json
import tempfile
import matplotlib.pyplot as plt
from torch.autograd import Variable
from torch.utils.data import DataLoader
from pycocotools.cocoeval import COCOeval
import time
from tqdm import tqdm
from dataload.cocodataset import *
from eval.evaluator import Evaluator
from utils.utils_coco import *
from utils.visualize import *

current_milli_time = lambda: int(round(time.time() * 1000))

class COCOEvaluator():
    """
    COCO AP Evaluation class.
    All the data in the val2017 dataset are processed \
    and evaluated by COCO API.
    """
    def __init__(self, data_dir, img_size, confthre, nmsthre):
        """
        Args:
            model_type (str): model name specified in config file
            data_dir (str): dataset root directory
            img_size (int): image size after preprocess. images are resized \
                to squares whose shape is (img_size, img_size).
            confthre (float):
                confidence threshold ranging from 0 to 1, \
                which is defined in the config file.
            nmsthre (float):
                IoU threshold of non-max supression ranging from 0 to 1.
        """
        self.classes = cfg.DATA["CLASSES"]
        self.val_data_path = cfg.DATA_PATH
        self.pred_result_path = os.path.join(cfg.PROJECT_PATH, 'prediction')
        self.__visual_imgs = cfg.TEST["NUM_VIS_IMG"]

        augmentation = {'LRFLIP': False, 'JITTER': 0, 'RANDOM_PLACING': False,
                        'HUE': 0, 'SATURATION': 0, 'EXPOSURE': 0, 'RANDOM_DISTORT': False}

        self.dataset = COCODataset(data_dir=data_dir,
                                   img_size=img_size,
                                   augmentation=augmentation,
                                   json_file=cfg.TEST["EVAL_JSON"],
                                   name=cfg.TEST["EVAL_NAME"])
        self.dataloader = DataLoader(self.dataset, batch_size=cfg.TEST["BATCH_SIZE"], shuffle=False,
                                     pin_memory=True, num_workers=cfg.TEST["NUMBER_WORKERS"])
        self.img_size = img_size
        self.confthre = confthre
        self.nmsthre = nmsthre
        self.inference_time = 0.
    def evaluate(self, model):
        """
        COCO average precision (AP) Evaluation. Iterate inference on the test dataset
        and the results are evaluated by COCO API.
        Args:
            model : model object
        Returns:
            ap50_95 (float) : calculated COCO AP for IoU=50:95
            ap50 (float) : calculated COCO AP for IoU=50
        """
        model.eval()
        cuda = torch.cuda.is_available()
        Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
        ids = []
        data_dict = []
        dataiterator = iter(self.dataloader)
        #print(" Val datasets number is : {}".format(len(self.dataloader)))
        for i in tqdm(range(len(self.dataloader))):
        #while True:
            #try:
            img, _, info_img, id_, img_path = next(dataiterator) # load a batch
            #except StopIteration:
                #break
            info_img = [float(info.numpy()) for info in info_img]
            id_ = int(id_)
            ids.append(id_)
            with torch.no_grad():
                img = Variable(img.type(Tensor))
                start_time = current_milli_time()
                _,outputs = model(img)
                self.inference_time += (current_milli_time() - start_time)
                outputs=outputs.unsqueeze(0)
                outputs = postprocess(
                    outputs, cfg.DATA["NUM"], self.confthre, self.nmsthre)
                if outputs[0] is None:
                    continue
                outputs = outputs[0].cpu().data

            for output in outputs:
                x1 = float(output[0])
                y1 = float(output[1])
                x2 = float(output[2])
                y2 = float(output[3])
                label = self.dataset.class_ids[int(output[6])]
                box = box2label((y1, x1, y2, x2), info_img)
                bbox = [box[1], box[0], box[3] - box[1], box[2] - box[0]]
                score = float(output[4].data.item() * output[5].data.item()) # object score * class score
                A = {"image_id": id_, "category_id": label, "bbox": bbox,
                     "score": score, "segmentation": []} # COCO json format
                data_dict.append(A)

            if self.__visual_imgs and i <= self.__visual_imgs:
                imgshow = cv2.imread(img_path[0])
                bboxes_prd = Evaluator(model).get_bbox(imgshow, cfg.TEST["MULTI_SCALE_TEST"], cfg.TEST["FLIP_TEST"])
                if bboxes_prd.shape[0] != 0:
                    boxes = bboxes_prd[..., :4]
                    class_inds = bboxes_prd[..., 5].astype(np.int32)
                    scores = bboxes_prd[..., 4]
                    visualize_boxes(image=imgshow, boxes=boxes, labels=class_inds, probs=scores, class_labels=self.classes)
                    path = os.path.join(self.pred_result_path, "imgs/{}.jpg".format(i))
                    cv2.imwrite(path, imgshow)


        annType = ['segm', 'bbox', 'keypoints']
        self.inference_time = 1.0 * self.inference_time / len(self.dataloader)
        # Evaluate the Dt (detection) json comparing with the ground truth
        if len(data_dict) > 0:
            cocoGt = self.dataset.coco
            _, tmp = tempfile.mkstemp()
            json.dump(data_dict, open(tmp, 'w'))
            cocoDt = cocoGt.loadRes(tmp)
            cocoEval = COCOeval(self.dataset.coco, cocoDt, annType[1])
            cocoEval.params.imgIds = ids
            #cocoEval.params.catIds = [0,1,2,3,4,5,6,7,8,9]
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()

            #if classwise:  # Compute per-category AP
                # Compute per-category AP
                # from https://github.com/facebookresearch/detectron2/
            precisions = cocoEval.eval['precision']
            self.cat_ids=[1,2,3,4,5,6,7,8,9,10]
            # precision: (iou, recall, cls, area range, max dets)
            assert len(self.cat_ids) == precisions.shape[2]

            results_per_category = []
            for idx, catId in enumerate(self.cat_ids):
                # area range index 0: all area ranges
                # max dets index -1: typically 100 per image
                #nm = self.coco.loadCats(catId)[0]
                precision = precisions[:, :, idx, 0, -1]
                precision = precision[precision > -1]
                if precision.size:
                    ap = np.mean(precision)
                else:
                    ap = float('nan')
                results_per_category.append(
                    (f'{float(ap):0.4f}'))
                print(results_per_category)


            '''
            # ----------pltshow------------- #
            # precision[t,:,k,a,m] PR curves recall-precision value
            # T:IoU thresh.5-.95, gap=0.05, t[0]=0.5,t[1]=0.55,t[2]=0.6,t[3]=0.65,t[4]=0.7,t[5]=0.75 ……,t[9]=0.95
            # R:101 recall thresh，0-101
            # K:class k[0] = person,k[1] = bycicle,.....COCO
            # A:area, a[0]=all,a[1]=small,a[2]=medium,a[3]=large
            # M:Maxdet m[0]=1,m[1]=10,m[2]=100

            #C75: PR at IoU=.75 (AP at strict IoU), area under curve corresponds to APIoU=.75 metric.
            #C50: PR at IoU=.50 (AP at PASCAL IoU), area under curve corresponds to APIoU=.50 metric.
            #Loc: PR at IoU=.10 (localization errors ignored, but not duplicate detections). All remaining settings use IoU=.1.
            #Sim: PR after supercategory false positives (fps) are removed. Specifically, any matches to objects with a different class label but that belong to the same supercategory don't count as either a fp (or tp). Sim is computed by setting all objects in the same supercategory to have the same class label as the class in question and setting their ignore flag to 1. Note that person is a singleton supercategory so its Sim result is identical to Loc.
            #Oth: PR after all class confusions are removed. Similar to Sim, except now if a detection matches any other object it is no longer a fp (or tp). Oth is computed by setting all other objects to have the same class label as the class in question and setting their ignore flag to 1.
            #BG: PR after all background (and class confusion) fps are removed. For a single category, BG is a step function that is 1 until max recall is reached then drops to 0 (the curve is smoother after averaging across categories).
            #FN: PR after all remaining errors are removed (trivially AP=1).

            pr_array1 = cocoEval.eval['precision'][0, :, 0, 0, 2]
            pr_array2 = cocoEval.eval['precision'][5, :, 0, 0, 2]
            #pr_array3 = cocoEval.eval['precision'][6, :, 0, 0, 2]
            #pr_array4 = cocoEval.eval['precision'][9, :, 0, 0, 2]
            x = np.arange(0.0, 1.01, 0.01)
            # x_1 = np.arange(0, 1.01, 0.111)
            plt.xlabel('IoU')
            plt.ylabel('precision')
            plt.xlim(0, 1.0)
            plt.ylim(0, 1.01)
            plt.grid(True)
            plt.plot(x, pr_array1, color='blue', linewidth = '3', label='IoU=0.5')
            plt.plot(x, pr_array2, color='green', linewidth = '3', label='IoU=0.75')
            plt.title("P-R curves catid=person maxDet=100")
            plt.legend(loc="lower left")
            plt.savefig("../prediction/APs.png", dpi=600)
            # plt.show()'''
            return cocoEval.stats[0], cocoEval.stats[1], self.inference_time
        else:
            return 0, 0, 0
