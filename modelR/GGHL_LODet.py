import sys
sys.path.append("..")
import torch.nn as nn
from modelR.backbones.mobilenetv2 import MobilenetV2
from modelR.necks.neck_lodet import Neck
from modelR.head.head_GGHL import Head
from utils.utils_basic import *
#from utils.prune_utils import *
class LODet_GGHL(nn.Module):
    def __init__(self, pre_weights=None):
        super(LODet_GGHL, self).__init__()
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__backnone = MobilenetV2(weight_path=pre_weights, extract_list=["6", "13", "conv"])#"17"
        self.__neck = Neck(fileters_in=[1280, 96, 32])
        # small
        self.__head_s = Head(nC=self.__nC, stride=self.__strides[0])
        # medium
        self.__head_m = Head(nC=self.__nC, stride=self.__strides[1])
        # large
        self.__head_l = Head(nC=self.__nC, stride=self.__strides[2])

    def forward(self, x):
        out = []
        x_s, x_m, x_l = self.__backnone(x)
        x_s, x_m, x_l = self.__neck(x_l, x_m, x_s)
        out.append(self.__head_s(x_s))
        out.append(self.__head_m(x_m))
        out.append(self.__head_l(x_l))
        if self.training:
            p, p_d = list(zip(*out))
            return p, p_d
        else:
            p, p_d = list(zip(*out))
            return p, torch.cat(p_d, 0)