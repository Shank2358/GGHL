import numpy as np

class CosineDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0., warmup=0):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        super(CosineDecayLR, self).__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup


    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (1 + np.cos(t/T_max * np.pi))
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    from matplotlib.ticker import FuncFormatter
    import math
    from modelR.GGHL import ABGH
    import torch.optim as optim
    import config.config as cfg

    net = ABGH()

    optimizer = optim.SGD(net.parameters(), cfg.TRAIN["LR_INIT"], cfg.TRAIN["MOMENTUM"], weight_decay=cfg.TRAIN["WEIGHT_DECAY"])
    #optimizer = optim.Adam(net.parameters(), lr = cfg.TRAIN["LR_INIT"])

    scheduler = CosineDecayLR(optimizer, math.ceil(cfg.TRAIN["EPOCHS"]/cfg.TRAIN["BATCH_SIZE"])*cfg.TRAIN["TRAIN_IMG_NUM"],
                              cfg.TRAIN["LR_INIT"], cfg.TRAIN["LR_END"], cfg.TRAIN["WARMUP_EPOCHS"]/cfg.TRAIN["BATCH_SIZE"]*cfg.TRAIN["TRAIN_IMG_NUM"])


    y = []
    for t in range(math.ceil(cfg.TRAIN["EPOCHS"]/cfg.TRAIN["BATCH_SIZE"])):
        for i in range(cfg.TRAIN["TRAIN_IMG_NUM"]):
            scheduler.step(cfg.TRAIN["TRAIN_IMG_NUM"]*t+i)
            y.append(optimizer.param_groups[0]['lr'])

    print(y)
    plt.figure()
    plt.plot(y, label='LambdaLR')
    plt.xlabel('steps')
    plt.ylabel('LR')
    plt.tight_layout()
    plt.savefig("../predictionR/lr.png", dpi=600)
    plt.show()