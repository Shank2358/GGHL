import sys
sys.path.append("..")
import torch.nn as nn
from modelR.backbones.darknet53 import Darknet53
from modelR.necks.neck_GGHLv2 import Neck
from modelR.head.head_GGHLv2_1 import Head1, Head2
from modelR.layers.convolutions import Convolutional
from utils.utils_basic import *

class GGHL(nn.Module):
    def __init__(self, init_weights=True, inputsize= int(cfg.TRAIN["TRAIN_IMG_SIZE"]), weight_path=None):
        super(GGHL, self).__init__()
        self.__strides = torch.FloatTensor(cfg.MODEL["STRIDES"])
        self.__nC = cfg.DATA["NUM"]
        self.__out_channel = self.__nC + 4 + 5 + 1
        self.__backnone = Darknet53()
        #self.__backnone = PVT2(weight_path=weight_path)
        #self.__fpn = Neck(fileters_in=[512, 320, 128, 64], fileters_out=self.__out_channel)
        self.__fpn = Neck(fileters_in=[1024, 512, 256, 128], fileters_out=self.__out_channel)
        self.__head1_s = Head1(filters_in=128, stride=self.__strides[0])
        self.__head1_m = Head1(filters_in=256, stride=self.__strides[1])
        self.__head1_l = Head1(filters_in=512, stride=self.__strides[2])

        self.__head2_s = Head2(filters_in=128, nC=self.__nC, stride=self.__strides[0])
        self.__head2_m = Head2(filters_in=256, nC=self.__nC, stride=self.__strides[1])
        self.__head2_l = Head2(filters_in=512, nC=self.__nC, stride=self.__strides[2])

        if init_weights:
            self.__init_weights()

    def forward(self, x):
        out = []
        x_8, x_16, x_32 = self.__backnone(x)
        loc2, cls2, loc1, cls1, loc0, cls0 = self.__fpn(x_32, x_16, x_8)
        x_s, x_s_de, offsets_loc_s, offsets_cls_s, mask_s = self.__head1_s(loc2)
        x_m, x_m_de, offsets_loc_m, offsets_cls_m, mask_m = self.__head1_m(loc1)
        x_l, x_l_de, offsets_loc_l, offsets_cls_l, mask_l = self.__head1_l(loc0)

        out_s, out_s_de = self.__head2_s(x_s_de, loc2, cls2, offsets_loc_s, offsets_cls_s, mask_s)
        out_m, out_m_de = self.__head2_m(x_m_de, loc1, cls1, offsets_loc_m, offsets_cls_m, mask_m)
        out_l, out_l_de = self.__head2_l(x_l_de, loc0, cls0, offsets_loc_l, offsets_cls_l, mask_l)

        out.append((x_s, x_s_de, out_s, out_s_de))
        out.append((x_m, x_m_de, out_m, out_m_de))
        out.append((x_l, x_l_de, out_l, out_l_de))

        if self.training:
            p1, p1_d, p2, p2_d = list(zip(*out))
            return p1, p1_d, p2, p2_d
        else:
            p1, p1_d, p2, p2_d = list(zip(*out))
            return p1, p1_d, p2, torch.cat(p2_d, 0)

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

if __name__ == '__main__':
    from modelR.get_model_complexity import get_model_complexity_info
    net = GGHL().cuda()
    flops, params = get_model_complexity_info(
        net, (800, 800), as_strings=False, print_per_layer_stat=True
    )
    print("GFlops: %.3fG" % (flops / 1e9))
    print("Params: %.2fM" % (params / 1e6))
