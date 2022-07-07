import argparse
import os
from modelR.GGHL import GGHL
import torch
from loguru import logger
import os
import config.config as cfg
import time
from utils.utils_coco import *
import argparse
import onnxruntime
import numpy as np
import onnx
def make_parser():
    parser = argparse.ArgumentParser("GGHL onnx Deployment")
    parser.add_argument('--onnx-filename',type=str,default='GGHL.onnx',help='the outname of onnx')
    parser.add_argument(
        "--input", default="images", type=str, help="input node name of onnx model"
    )
    parser.add_argument(
        "--output", default="output", type=str, help="(●'◡'●)"
    )
    parser.add_argument('--output2',type=str,default='( •̀ ω •́ )y',help='Demo')
    parser.add_argument('--dynamic',action='store_true',help='whether the input shape should be dynamic or not')
    parser.add_argument('--log_path', type=str, default='log/deploy', help='log path')
    parser.add_argument('--weight_path', type=str, default='weights/GGHL_darknet53_fpn3_DOTA_76.95.pt', help='weight file path')
    parser.add_argument('--opset',default=15,type=int,help='onnx opset version')
    parser.add_argument('--simonnx',default=False,type=bool,help='onnx simplify')
    return parser
@logger.catch
def main():
    args = make_parser().parse_args()
    logger.info("args value: {}".format(args))
    
    print("loading weight file from :{}".format(args.weight_path))
    model = GGHL().eval()
    ckpt = torch.load(os.path.join(args.weight_path),map_location='cpu')
    model.load_state_dict(ckpt)
    logger.info("loading checkpoint done")
    dummy_input = torch.randn(1,3,cfg.TEST["TEST_IMG_SIZE"],cfg.TEST["TEST_IMG_SIZE"])
    torch.onnx.export(model,
                      dummy_input,
                      args.onnx_filename,
                      input_names=[args.input],
                      output_names=[args.output,args.output2,'demo','output'],
                      opset_version = 11)
    # Network has dynamic or shape inputs, but no optimization profile has been defined.
    # 将onnx导入之后，需要剔除三个输出节点
    model_onnx = onnx.load(args.onnx_filename)
    graph = model_onnx.graph
    out = graph.output
    for i in range(len(graph.output)-1):
        graph.output.remove(out[0])
    #去除掉上面三个的头节点，转换成
    onnx.checker.check_model(model_onnx)
    onnx.save(model_onnx,args.onnx_filename)
    logger.info("generate onnx name {}".format(args.onnx_filename))

        
    
    
if __name__ =='__main__':
    main()   
    