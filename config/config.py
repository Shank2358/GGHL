# coding=utf-8
DATA_PATH = "/opt/datasets/DOTA/"#"F:/Datasets/DOTAx/test/"#
PROJECT_PATH = "./"

DATA = {"CLASSES": ['plane',
                    'baseball-diamond',
                    'bridge',
                    'ground-track-field',
                    'small-vehicle',
                    'large-vehicle',
                    'ship',
                    'tennis-court',
                    'basketball-court',
                    'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'],
        "NUM": 15}

DATASET_NAME = "train_DOTA_angle"
MODEL = {"STRIDES":[8, 16, 32]}

MAX_LABEL = 1000
SHOW_HEATMAP = False
SCALE_FACTOR = 2.0#2.0
SCALE_FACTOR_A = 2.0 #3.0

TRAIN = {
    "EVAL_TYPE": 'VOC',  # ['VOC', 'COCO']
    "TRAIN_IMG_SIZE": 800,
    "TRAIN_IMG_NUM": 63182,
    "AUGMENT": True,
    "MULTI_SCALE_TRAIN": True,
    "MULTI_TRAIN_RANGE": [23, 28, 1],
    "BATCH_SIZE": 32,
    "IOU_THRESHOLD_LOSS": 0.6,
    "EPOCHS": 101,
    "NUMBER_WORKERS": 12,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 1e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 5,
    "IOU_TYPE": 'GIOU'  # ['GIOU','CIOU']
}

TEST = {
    "EVAL_TYPE": 'VOC',
    "EVAL_JSON": 'test.json',
    "EVAL_NAME": 'test',
    "NUM_VIS_IMG": 0,
    "TEST_IMG_SIZE": 800,
    "BATCH_SIZE": 4,
    "NUMBER_WORKERS": 24,
    "CONF_THRESH": 0.2,
    "NMS_THRESH": 0.45,
    "IOU_THRESHOLD": 0.5,
    "NMS_METHODS": 'NMS',
    "MULTI_SCALE_TEST": False,
    "MULTI_TEST_RANGE": [736, 864, 96],
    "FLIP_TEST": False
}






