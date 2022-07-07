from distutils.command.build import build
from lib2to3.pgen2.parse import Parser
from threading import local
from matplotlib.style import context
import numpy as np
import tensorrt as trt
import onnx
import numpy
import os
import argparse
import sys
import config.config as cfg
import time
import torch
from utils.utils_basic import py_cpu_nms_poly_fast
import cv2 as cv2
import numpy as np
from collections import namedtuple,OrderedDict
import torch.nn as nn
import pycuda.driver as cuda
import pycuda.autoinit
import onnx
import onnxruntime
from modelR.GGHL import GGHL
import math
sys.path.insert(0,os.getcwd())
def make_parse():
    parser = argparse.ArgumentParser('GGHL TensorRT Deployment')
    parser.add_argument('--use_half', default=False, type=bool,help='FP16 半精度运算')
    parser.add_argument('--use_INT8', default=False, type=bool,help='QAT quantization')
    parser.add_argument(
        '--onnx_path', default='/home/crescent/GGHL/GGHL.onnx', type=str, help='IR 中间表达')
    parser.add_argument('--engine_path',default='/home/crescent/GGHL/GGHL.engine', type=str, help='Engine')
    parser.add_argument('--onnx-filename', type=str,
                        default='GGHL.onnx', help='the outname of onnx')
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="(●'◡'●)"
    )
    parser.add_argument('--output2', type=str,
                        default='( •̀ ω •́ )y', help='Demo')
    parser.add_argument('--dynamic', action='store_true',
                        help='whether the input shape should be dynamic or not')
    parser.add_argument('--log_path', type=str,
                        default='log/deploy', help='log path')
    parser.add_argument('--weight_path', type=str,
                        default='weights/GGHL_darknet53_fpn3_DOTA_76.95.pt', help='weight file path')
    parser.add_argument('--opset', default=11, type=int,
                        help='onnx opset version')
    return parser
class HostDeviceMem(object):
    # 将内存中Host输入数据转存到对应的显存中，将显存中的结果保存到内充中
    # 每个输入张量与输出变量，需要分配两块资源，分别是内存中
    def __init__(self,host_mem,device_mem):
        self.host = host_mem
        self.device = device_mem
    def __str__(self) -> str:
        return "Hosy:\n"+str(self.host)+"\n Device:\n"+str(self.device)
    def __repr__(self):
        return self.__str__()


def non_max_suppression_4points(
                                prediction, conf_thres=0.2, iou_thres=0.45, merge=False, classes=None, multi_label=False, agnostic=False,
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
    print(prediction[0].shape)
    nc = prediction[0].shape[1] - 9
    # class_index = nc + 9
    # xc : (batch_size, num_boxes) 对应位置为1说明该box超过置信度
    xc = prediction[..., 8] > conf_thres  # candidates

    # Settings
    # (pixels) minimum and maximum box width and height
    min_wh, max_wh = 2, 4096
    max_det = 500  # maximum number of detections per image
    time_limit = 10.0  # seconds to quit after
    redundant = True  # require redundant detections 要求冗余检测
    multi_label &= nc > 1  # multiple labels per box (adds 0.5ms/img)

    t = time.time()
      # output: (batch_size, ?)
    
    output = [torch.zeros(
           (0, 10), device=prediction.device)] * prediction.shape[0]
    for xi, x in enumerate(prediction):  # image index, image inference
            # 取出数组中索引为True的的值即将置信度符合条件的boxes存入x中   x -> (num_confthres_boxes, no)
            x = x[xc[xi]]
            # If none remain process next image
            if not x.shape[0]:
                continue
            # conf = obj_conf * cls_conf
            x[:, 9:] = torch.pow(x[:, 9:], 0.45) * \
                torch.pow(x[:, 8:9], 0.55)
            box = x[:, :8]
            if multi_label:
                print('multi_label')
                i, j = (x[:, 9:] > math.sqrt(conf_thres) /
                        2).nonzero(as_tuple=False).T
                x = torch.cat(
                    (box[i], x[i, j + 9, None], j[:, None].float()), 1)
            else:  # best class only
                conf, j = x[:, 9:].max(1, keepdim=True)
                x = torch.cat((box, conf, j.float()), 1)[
                    conf.view(-1) > conf_thres]
                print(j.shape,'j')
                print(j)
                print(conf.shape,'conf')
            if without_iouthres:  # 不做nms_iou
                output[xi] = x
                continue
            # Filter by class 按类别筛选
            print(x[:,9])
            if classes:
                # any（1）函数表示每行满足条件的返回布尔值
                x = x[(x[:, 9:] == torch.tensor(classes, device=x.device)).any(1)]
            # If none remain process next image
            n = x.shape[0]  # number of boxes
            if not n:
                continue
            # Sort by confidence
            c = x[:, 9:] * (0 if agnostic else max_wh)  # classes
            print('CCCC',c.shape)
            print(c)
            boxes_4points, scores = x[:, :8] , x[:, 8]
            print(boxes_4points)
            print("scores",scores.shape,"Bboxes_4_point",boxes_4points.shape)
            i = np.array(py_cpu_nms_poly_fast(
                np.double(boxes_4points.detach().cpu().numpy()), scores.detach().cpu().numpy(), iou_thres))
            if i.shape[0] > max_det:  # limit detections
                i = i[:max_det]
            temp = x[i].clone()
            # output[xi] = x[i]  # 根据nms索引提取x中的框  x.size=(num_conf_nms, [xywhθ,conf,classid]) θ∈[0,179]
            output[xi] = temp
            if (time.time() - t) > time_limit:
                break  # time limit exceeded
    return output
def GiB(val):
    return val * 1 << 30
class TensorEngine(nn.Module):
    def __init__(self,args,device = torch.device('cuda:0')):
        super().__init__()
        self.args = args
        self.device = device 
        self.build_engine(args.onnx_path,args.use_half,args.engine_path) 
    def build_engine(self,onnx_path,use_half,engine_path):
        """
            转换模型是onnx格式，未经简化过后的模型

        Args:
            onnx_path (_type_): onnx中间表达
            using_haif (_type_): 半精度浮点数
        """
        logger = trt.Logger(trt.Logger.INFO)
        if os.path.exists(engine_path):
            with open(engine_path, "rb") as f:
                serialized_engine = f.read()
        else:
            builder = trt.Builder(logger)
            network= builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))
            parser = trt.OnnxParser(network,logger)
            success = parser.parse_from_file(onnx_path)
            for idx in range(parser.num_errors):
                print(parser.get_error(idx))
            if not success:
                raise ValueError
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 10 << 30)
            # 默认是FP32的处理方法
            if use_half:
                config.set_flag(trt.BuilderFlag.FP16)
                # 单精度浮点
            serialized_engine = builder.build_serialized_network(network, config)
            with open(os.path.join(os.getcwd(),'GGHL.engine'),'wb') as f:
                f.write(serialized_engine)
        runtime = trt.Runtime(logger)
        engine = runtime.deserialize_cuda_engine(serialized_engine)
        self.__dict__.update(locals())
    def forward(self,image):
        context = self.engine.create_execution_context()
        context.set_binding_shape(0, (1, 3, 800,800))
        GGHL_inputs, GGHL_outputs, GGHL_bindings = TensorEngine.allocate_buffers(
            self.engine, True)
        stream = cuda.Stream()
        
        GGHL_inputs[0].host = image.float().numpy()
        [cuda.memcpy_htod_async(inp.device, inp.host, stream)
         for inp in GGHL_inputs]
        stream.synchronize()
        context.execute_async_v2(
            bindings=GGHL_bindings, stream_handle=stream.handle)
        stream.synchronize()
        [cuda.memcpy_dtoh_async(out.host, out.device, stream)
         for out in GGHL_outputs]
        stream.synchronize()
        pred = np.array(GGHL_outputs[-1].host)     
        return pred.reshape(-1,29)
    @staticmethod
    def allocate_buffers(engine,is_explicit_batch=False,dynamic_shapes=[]):
        inputs = []
        outputs = []
        bindings = []
        for binding in engine:
            dims = engine.get_binding_shape(binding)
            print(dims)
            if dims[0] == -1:
                assert(len(dynamic_shapes) > 0)
                dims[0] = dynamic_shapes[0]
            size = trt.volume(dims) * 1
            print(size)
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            device_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(device_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDeviceMem(host_mem, device_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, device_mem))
        return inputs, outputs,bindings

          

def xywh2xyxy(x):
    # Convert bounding box format from [x, y, w, h] to [x1, y1, x2, y2]
    y = torch.zeros_like(x) if isinstance(
        x, torch.Tensor) else np.zeros_like(x)
    y[:, 0] = x[:, 0] - x[:, 2] / 2
    y[:, 1] = x[:, 1] - x[:, 3] / 2
    y[:, 2] = x[:, 0] + x[:, 2] / 2
    y[:, 3] = x[:, 1] + x[:, 3] / 2
    return y
def convert_pred(pred_bbox, test_input_size, org_img_shape, valid_scale):
    pred_xyxy = xywh2xyxy(pred_bbox[:, :4])  # xywh2xyxy
    pred_conf = pred_bbox[:, 13]
    pred_prob = pred_bbox[:, 14:]
    org_h, org_w = org_img_shape
    resize_ratio = min(1.0 * test_input_size / org_w,
                       1.0 * test_input_size / org_h)
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

    invalid_mask = torch.logical_or(
        (pred_xyxy[:, 0] > pred_xyxy[:, 2]), (pred_xyxy[:, 1] > pred_xyxy[:, 3]))
    pred_xyxy[invalid_mask] = 0
    pred_s[invalid_mask] = 0
    # (4)去掉不在有效范围内的bbox
    bboxes_scale = torch.sqrt(
        (pred_xyxy[..., 2:3] - pred_xyxy[..., 0:1]) * (pred_xyxy[..., 3:4] - pred_xyxy[..., 1:2]))
    scale_mask = torch.logical_and(
        (valid_scale[0] < bboxes_scale), (bboxes_scale < valid_scale[1])).squeeze(-1)
    print(scale_mask.shape,'scale_mask')
    # (5)将score低于score_threshold的bbox去掉
    classes = torch.argmax(pred_prob, dim=-1)
    scores = pred_conf * pred_prob[torch.arange(len(pred_xyxy)), classes]
    score_mask = scores > cfg.TEST["CONF_THRESH"]
    mask = torch.logical_and(scale_mask, score_mask)
    print(classes.shape,'classes_shape')
    print(scores.shape,'scores_shape')
    
    pred_xyxy = pred_xyxy[mask]
    pred_s = pred_s[mask]

    pred_conf = pred_conf[mask]

    #classes = classes[mask]
    pred_prob = pred_prob[mask]

    x1 = pred_s[:, 0:1] * (pred_xyxy[:, 2:3] -
                           pred_xyxy[:, 0:1]) + pred_xyxy[:, 0:1]
    y1 = pred_xyxy[:, 1:2]
    x2 = pred_xyxy[:, 2:3]
    y2 = pred_s[:, 1:2] * (pred_xyxy[:, 3:4] -
                           pred_xyxy[:, 1:2]) + pred_xyxy[:, 1:2]
    x3 = pred_xyxy[:, 2:3] - pred_s[:, 2:3] * \
        (pred_xyxy[:, 2:3] - pred_xyxy[:, 0:1])
    y3 = pred_xyxy[:, 3:4]
    x4 = pred_xyxy[:, 0:1]
    y4 = pred_xyxy[:, 3:4] - pred_s[:, 3:4] * \
        (pred_xyxy[:, 3:4] - pred_xyxy[:, 1:2])
    coor4points = torch.cat([x1, y1, x2, y2, x3, y3, x4, y4], dim=-1)

    bboxes = torch.cat(
        [coor4points, pred_conf.unsqueeze(-1), pred_prob], dim=-1)

    #print(bboxes)
    bs = 1  # cfg.TEST["NUMBER_WORKERS"]
    #_, out_num = bboxes.shape[1]
    #print(bboxes.shape)
    bboxes = bboxes.view(bs, -1, bboxes.shape[1])
    return bboxes


if __name__ == '__main__':
    args = make_parse().parse_args()
    model = TensorEngine(args=args)
   # ort_session = onnxruntime.InferenceSession("/home/crescent/GGHL/GGHL.onnx")
    
    ori_img = cv2.imread("/home/crescent/images/P0035__1024__0___184.png")
    img = np.copy(ori_img)
    org_h,org_w,_ = np.array(ori_img).shape
    print(org_h)
    img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB).astype(np.float32)
    img = np.array(cv2.resize(img,(800,800)))/255.0
    img = img.transpose(2,0,1)
    img = img[np.newaxis,...]
    img = np.ascontiguousarray(img, np.float32)
    print(img.shape)
       # img = torch.from_numpy(img[np.newaxis,...]).float().to(torch.device('cpu'))
    """
    model = GGHL().eval().to(torch.device('cpu'))
    ckpt = torch.load(os.path.join(args.weight_path), map_location=torch.device('cpu'))
    model.load_state_dict(ckpt)
    with torch.no_grad():
        _,output = model(img)
    output = output.squeeze()
    """
        
       
    pred_bbox= model(torch.from_numpy(img).float())
  #  pred_bbox = np.array(output[-1])
    pred_bbox_tensor = torch.from_numpy(pred_bbox).float()
    print(org_h,'WWQ')
    bboxes = convert_pred(pred_bbox_tensor,800,(org_h,org_w),(0,np.inf))
    print(bboxes.shape,'final')
    bboxes = non_max_suppression_4points(
        bboxes, 0.10, 0.45)[0].detach().cpu().numpy()
    
    for bbox in bboxes:
        x1 = bbox[0]  # ]a_rota[0] * (coor[2] - coor[0]) + coor[0]
        y1 = bbox[1]
        x2 = bbox[2]
        y2 = bbox[3]
        x3 = bbox[4]
        y3 = bbox[5]
        x4 = bbox[6]
        y4 = bbox[7]
        score = bbox[8]
        class_ind = int(bbox[9])
        class_name = cfg.DATA["CLASSES"][class_ind]
        #print(class_name, score, str(int(x1)), str(int(y1)), str(int(x2)), str(int(y2)), str(int(x3)), str(int(y3)), str(int(x4)), str(int(y4)))
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
        elif int(class_ind) == 14:  color = (255, 128, 255)
        cv2.polylines(ori_img, [points], 1, color, 2)
        font = cv2.FONT_HERSHEY_SIMPLEX
        ori_img = cv2.putText(ori_img, class_name + ' ' + score[:4], (int(
            float(x1)), int(float(y1))), font, 0.3, (255, 255, 255), 1)
    cv2.imwrite('QAQ4.png', ori_img, [int(cv2.IMWRITE_JPEG_QUALITY), 100])



        
