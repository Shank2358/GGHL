import logging
import argparse
import torch.optim as optim
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
import utils.gpu as gpu
from utils import cosine_lr_scheduler
from utils.log import Logger
import dataloadR.datasets_obb as data
from dataloadR.batch_sampler import BatchSampler, RandomSampler
from modelR.GGHL import GGHL
from modelR.loss.loss_jol import Loss
from evalR.evaluatorGGHL import *
from torch.cuda import amp
import torch.backends.cudnn as cudnn

import cv2
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

def split_parameters(module):
    params_decay = []
    params_no_decay = []
    for m in module.modules():
        if isinstance(m, torch.nn.Linear):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.conv._ConvNd):
            params_decay.append(m.weight)
            if m.bias is not None:
                params_no_decay.append(m.bias)
        elif isinstance(m, torch.nn.modules.batchnorm._BatchNorm):
            params_no_decay.extend([*m.parameters()])
        elif isinstance(m, torch.nn.LayerNorm):
            params_no_decay.extend([*m.parameters()])
        elif len(list(m.children())) == 0:
            params_decay.extend([*m.parameters()])
    #print("aaa", len(params_decay), len(params_no_decay))
    assert len(list(module.parameters())) == len(params_decay) + len(params_no_decay)
    return params_decay, params_no_decay

class Trainer(object):
    def __init__(self,  weight_path, resume, gpu_id):
        init_seeds(0)
        self.device = gpu.select_device(gpu_id)
        print(self.device)
        self.cuda = self.device.type != 'cpu'
        self.start_epoch = 0
        self.best_mAP = 0.
        self.epochs = cfg.TRAIN["EPOCHS"]
        self.weight_path = weight_path
        self.multi_scale_train = cfg.TRAIN["MULTI_SCALE_TRAIN"]
        if self.multi_scale_train: print('Using multi scales training')
        else: print('train img size is {}'.format(cfg.TRAIN["TRAIN_IMG_SIZE"]))

        self.train_dataset = data.Construct_Dataset(anno_file_name=cfg.DATASET_NAME, img_size=cfg.TRAIN["TRAIN_IMG_SIZE"])
        self.train_dataloader = DataLoader(self.train_dataset,
                                           batch_sampler=BatchSampler(RandomSampler(self.train_dataset),
                                                                      batch_size=cfg.TRAIN["BATCH_SIZE"],
                                                                      drop_last=True,
                                                                      multiscale_step=10,
                                                                      img_sizes=list(
                                                                          range(
                                                                              cfg.TRAIN["MULTI_TRAIN_RANGE"][0] * 32,
                                                                              cfg.TRAIN["MULTI_TRAIN_RANGE"][1] * 32,
                                                                              cfg.TRAIN["MULTI_TRAIN_RANGE"][2] * 32))
                                                                      ),
                                           num_workers=cfg.TRAIN["NUMBER_WORKERS"],
                                           #shuffle=True,
                                           pin_memory=True)

        net_model = GGHL(weight_path=self.weight_path)
        self.model = net_model.to(self.device)
        #torch.backends.cudnn.benchmark = False


        # Optimizer
        g0, g1, g2 = [], [], []  # optimizer parameter groups
        for v in self.model.modules():
            if hasattr(v, 'bias') and isinstance(v.bias, torch.nn.Parameter):  # bias
                g2.append(v.bias)
            if isinstance(v, torch.nn.BatchNorm2d):  # weight (no decay)
                g0.append(v.weight)
            elif hasattr(v, 'weight') and isinstance(v.weight, torch.nn.Parameter):  # weight (with decay)
                g1.append(v.weight)

        # if opt.adam:
        #     self.optimizer = optim.Adam(g0, lr=cfg.TRAIN["LR_INIT"], betas=(cfg.TRAIN["MOMENTUM"], 0.999))  # adjust beta1 to momentum
        # else:
        self.optimizer = optim.SGD(g0, lr=cfg.TRAIN["LR_INIT"], momentum=cfg.TRAIN["MOMENTUM"], nesterov=True)

        self.optimizer.add_param_group({'params': g1, 'weight_decay': cfg.TRAIN["WEIGHT_DECAY"]})  # add g1 with weight_decay
        self.optimizer.add_param_group({'params': g2})  # add g2 (biases)
        del g0, g1, g2
        self.__load_optimizer_weights(weight_path, resume)

        self.criterion = Loss()

        self.__load_model_weights(weight_path, resume)

        self.scheduler = cosine_lr_scheduler.CosineDecayLR(self.optimizer,
                                                           T_max=self.epochs*len(self.train_dataloader),
                                                           lr_init=cfg.TRAIN["LR_INIT"],
                                                           lr_min=cfg.TRAIN["LR_END"],
                                                           warmup=cfg.TRAIN["WARMUP_EPOCHS"] * len(self.train_dataloader))
        self.scaler = amp.GradScaler(enabled=self.cuda)

    def __load_optimizer_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(
                os.path.split(weight_path)[0], "last.pt"
            )  # backup_epoch30
            chkpt = torch.load(last_weight, map_location=self.device)
            #self.start_epoch = 0#chkpt["epoch"] + 1
            if chkpt["optimizer"] is not None:
                self.optimizer.load_state_dict(chkpt["optimizer"])
                #print(chkpt["optimizer"])
                self.best_mAP = chkpt["best_mAP"]
            del chkpt

    def __load_model_weights(self, weight_path, resume):
        if resume:
            last_weight = os.path.join(
                os.path.split(weight_path)[0], "last.pt"
            )
            chkpt = torch.load(last_weight, map_location=self.device)

            self.model.load_state_dict(chkpt["model"])  # , False
            self.start_epoch = 0  # chkpt["epoch"] + 1
            del chkpt  # , model_dict
        else:
            self.model.load_darknet_weights(weight_path)

    def __save_model_weights(self, epoch, mAP):
        if mAP > self.best_mAP:
            self.best_mAP = mAP
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        last_weight = os.path.join(os.path.split(self.weight_path)[0], "last.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt, last_weight)

        if self.best_mAP == mAP:
            torch.save(chkpt['model'], best_weight)
        if epoch > 0 and epoch % 5 == 0:
            torch.save(chkpt, os.path.join(os.path.split(self.weight_path)[0], 'backup_epoch%g.pt'%epoch))
            #
        del chkpt

    def __save_model_weights_best(self, epoch):
        best_weight = os.path.join(os.path.split(self.weight_path)[0], "best.pt")
        chkpt = {'epoch': epoch,
                 'best_mAP': self.best_mAP,
                 'model': self.model.state_dict(),
                 'optimizer': self.optimizer.state_dict()}
        torch.save(chkpt['model'], best_weight)
        del chkpt

    def train(self):
        global writer
        #logger.info(self.model)
        logger.info(" Training start!  Img size:{:d},  Batchsize:{:d},  Number of workers:{:d}".format(
            cfg.TRAIN["TRAIN_IMG_SIZE"], cfg.TRAIN["BATCH_SIZE"], cfg.TRAIN["NUMBER_WORKERS"]))
        logger.info(" Train datasets number is : {}".format(len(self.train_dataset)))
        for epoch in range(self.start_epoch, self.epochs):
            start = time.time()
            self.model.train()
            mloss = torch.zeros(10)

            for i, (imgs, label_sbbox, label_mbbox, label_lbbox) in enumerate(self.train_dataloader):

                self.scheduler.step(len(self.train_dataloader)*epoch + i)
                imgs = imgs.to(self.device)
                with amp.autocast(enabled=self.cuda):
                    p, p_d = self.model(imgs)
                    label_sbbox = label_sbbox.to(self.device)
                    label_mbbox = label_mbbox.to(self.device)
                    label_lbbox = label_lbbox.to(self.device)
                    loss, loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_s, loss_r, loss_l = self.criterion(p, p_d, label_sbbox, label_mbbox, label_lbbox, epoch, i)

                self.scaler.scale(loss).backward()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

                loss_items = 10 * torch.tensor([loss_fg, loss_bg, loss_pos, loss_neg, loss_iou, loss_cls, loss_s, loss_r, loss_l, loss])
                mloss = (mloss * i + loss_items) / (i + 1)
                mAP = 0

                if i % 50 == 0:
                    logger.info(
                        " Epoch:[{:3}/{}] Batch:[{:3}/{}] Img_size:[{:3}] Loss:{:.4f} "
                        "Loss_fg:{:.4f} | Loss_bg:{:.4f} | Loss_pos:{:.4f} | Loss_neg:{:.4f} "
                        "| Loss_iou:{:.4f} | Loss_cls:{:.4f} | Loss_s:{:.4f} | Loss_r:{:.4f} | "
                        "Loss_l:{:.4f} | LR:{:g}".format(
                            epoch, self.epochs, i, len(self.train_dataloader) - 1, imgs.size(-1),
                            mloss[9], mloss[0], mloss[1], mloss[2], mloss[3],
                            mloss[4], mloss[5], mloss[6], mloss[7], mloss[8 ],
                            self.optimizer.param_groups[0]['lr']
                        ))
                    writer.add_scalar('loss_fg', mloss[0], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_bg', mloss[1], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_pos', mloss[2], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_neg', mloss[3], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_iou', mloss[4], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_cls', mloss[5], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_s', mloss[6], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_r', mloss[7], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('loss_l', mloss[8], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)
                    writer.add_scalar('train_loss', mloss[9], len(self.train_dataloader)
                                      * (cfg.TRAIN["BATCH_SIZE"]) * epoch + i)

                # if self.multi_scale_train and (i+1) % 10 == 0:
                #     self.train_dataset.img_size = random.choice(range(
                #         cfg.TRAIN["MULTI_TRAIN_RANGE"][0], cfg.TRAIN["MULTI_TRAIN_RANGE"][1],
                #         cfg.TRAIN["MULTI_TRAIN_RANGE"][2])) * 32
            if epoch >= 50 and epoch % 5 == 0 and cfg.TRAIN["EVAL_TYPE"] == 'VOC':
                logger.info("===== Validate =====".format(epoch, self.epochs))
                with torch.no_grad():
                    APs, inference_time = Evaluator(self.model).APs_voc()
                    for i in APs:
                        logger.info("{} --> mAP : {}".format(i, APs[i]))
                        mAP += APs[i]
                    mAP = mAP / self.train_dataset.num_classes
                    logger.info("mAP : {}".format(mAP))
                    logger.info("inference time: {:.2f} ms".format(inference_time))
                    writer.add_scalar('mAP', mAP, epoch)
            self.__save_model_weights(epoch, mAP)
            logger.info('Save weights Done')
            logger.info("mAP: {:.3f}".format(mAP))
            end = time.time()
            logger.info("Inference time: {:.4f}s".format(end - start))

        logger.info("Training finished.  Best_mAP: {:.3f}%".format(self.best_mAP))

os.environ['KMP_DUPLICATE_LIB_OK']='True'
if __name__ == "__main__":
    global logger, writer
    parser = argparse.ArgumentParser()
    parser.add_argument('--weight_path', type=str, default='weight/darknet53_448.weights', help='weight file path')
    parser.add_argument('--resume', action='store_true', default=True, help='resume training flag')
    parser.add_argument('--gpu_id', type=int, default=0, help='gpu id')
    parser.add_argument('--log_path', type=str, default='log/', help='log path')
    opt = parser.parse_args()
    writer = SummaryWriter(logdir=opt.log_path + '/event')
    logger = Logger(log_file_name=opt.log_path + '/log.txt', log_level=logging.DEBUG, logger_name='GGHL').get_log()
    Trainer(weight_path=opt.weight_path, resume=opt.resume, gpu_id=opt.gpu_id).train()
    #tensorboard --logdir=log
