import sys
sys.path.append("..")
import torch.nn as nn
from modelR.backbones.darknet53_4 import Darknet53
from modelR.necks.neck_YOLO import FPN_YOLOV3
from modelR.head.head_AnchorBase import Yolo_head
from modelR.layers.convolutions import Convolutional
from utils.utils_basic import *

class Yolov3R(nn.Module):
    """
    Note ： int the __init__(), to define the modules should be in order, because of the weight file is order
    """
    def __init__(self, init_weights=True, inputsize= int(cfg.TRAIN["TRAIN_IMG_SIZE"])):
        super(Yolov3R, self).__init__()
        self.anchor_fact = inputsize/608
        self.__anchors = torch.FloatTensor(cfg.MODEL["ANCHORS"])
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = cfg.MODEL["ANCHORS_PER_SCLAE"] * (self.__nC + 5 + 5)###########xywhc,a1-a4,r
        #self.__backnone = Darknet53()

        self.__backnone = Darknet53()#Darknet53_NPAttention()#Darknet53()
        #self.__backnone = ResNeSt() #Darknet53_NPAttention()fileters_in=[1024, 512, 256]
        self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256],
                                fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
        #self.__fpn = MSR_FPN_Lite(fileters_in=[1024, 512, 256])
            #,fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])

        #self.__fpn = FPN_YOLOV3(fileters_in=[1024, 512, 256], fileters_out=[self.__out_channel, self.__out_channel, self.__out_channel])
        self.__head_s = Yolo_head(nC=self.__nC, anchors=self.__anchors[0], stride=self.__strides[0], fact = self.anchor_fact)
        self.__head_m = Yolo_head(nC=self.__nC, anchors=self.__anchors[1], stride=self.__strides[1], fact = self.anchor_fact)
        self.__head_l = Yolo_head(nC=self.__nC, anchors=self.__anchors[2], stride=self.__strides[2], fact = self.anchor_fact)
        if init_weights:
            self.__init_weights()

    def forward(self, x):
        out = []
        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l = self.__fpn(x_l, x_m, x_s)
        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d  # smalll, medium, large
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)


    def __init_weights(self):
        " Note ：nn.Conv2d nn.BatchNorm2d'initing modes are uniform "
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.normal_(m.weight.data, 0.0, 0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))
            elif isinstance(m, nn.BatchNorm2d):
                torch.nn.init.constant_(m.weight.data, 1.0)
                torch.nn.init.constant_(m.bias.data, 0.0)
                print("initing {}".format(m))
            elif isinstance(m, nn.Linear):
                m.weight.data.normal_(0,0.01)
                if m.bias is not None:
                    m.bias.data.zero_()
                print("initing {}".format(m))

    def load_darknet_weights(self, weight_file, cutoff=52):
        "https://github.com/ultralytics/yolov3/blob/master/models.py"
        print("load darknet weights : ", weight_file)
        with open(weight_file, 'rb') as f:
            _ = np.fromfile(f, dtype=np.int32, count=5)
            weights = np.fromfile(f, dtype=np.float32)
        count = 0
        ptr = 0
        for m in self.modules():
            if isinstance(m, Convolutional):
                # only initing backbone conv's weights
                if count == cutoff:
                    break
                count += 1
                conv_layer = m._Convolutional__conv
                if m.norm == "bn":
                    # Load BN bias, weights, running mean and running variance
                    bn_layer = m._Convolutional__norm
                    num_b = bn_layer.bias.numel()  # Number of biases
                    # Bias
                    bn_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.bias.data)
                    bn_layer.bias.data.copy_(bn_b)
                    ptr += num_b
                    # Weight
                    bn_w = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.weight.data)
                    bn_layer.weight.data.copy_(bn_w)
                    ptr += num_b
                    # Running Mean
                    bn_rm = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_mean)
                    bn_layer.running_mean.data.copy_(bn_rm)
                    ptr += num_b
                    # Running Var
                    bn_rv = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(bn_layer.running_var)
                    bn_layer.running_var.data.copy_(bn_rv)
                    ptr += num_b
                    print("loading weight {}".format(bn_layer))
                else:
                    # Load conv. bias
                    num_b = conv_layer.bias.numel()
                    conv_b = torch.from_numpy(weights[ptr:ptr + num_b]).view_as(conv_layer.bias.data)
                    conv_layer.bias.data.copy_(conv_b)
                    ptr += num_b
                # Load conv. weights
                num_w = conv_layer.weight.numel()
                conv_w = torch.from_numpy(weights[ptr:ptr + num_w]).view_as(conv_layer.weight.data)
                conv_layer.weight.data.copy_(conv_w)
                ptr += num_w
                print("loading weight {}".format(conv_layer))
