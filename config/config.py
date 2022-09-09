# coding=utf-8
DATA_PATH = "F:/Datasets/HRSC2016/test/"
#DATA_PATH = "F:/Datasets/DOTAx/test/"#
#DATA_PATH = 'F:/Datasets/DroneVehicle/test/'
#DATA_PATH = "/opt/datasets/DOTA/"
PROJECT_PATH = "./"

# DATA = {"CLASSES": ['car', 'truck', 'bus', 'van', 'freight car'],
#         "NUM": 5}

# DATA = {"CLASSES": ['plane',
#                     'baseball-diamond',
#                     'bridge',
#                     'ground-track-field',
#                     'small-vehicle',
#                     'large-vehicle',
#                     'ship',
#                     'tennis-court',
#                     'basketball-court',
#                     'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'],
#         "NUM": 15}

DATA = {"CLASSES": ['ship'], "NUM": 1}

DATASET_NAME = "train_HRSC2016"
MODEL = {"STRIDES":[8, 16, 32]}

MAX_LABEL = 1000
SHOW_HEATMAP = False
SCALE_FACTOR = 2.0
SCALE_FACTOR_A = 2.0

TRAIN = {
    "EVAL_TYPE": 'VOC',
    "TRAIN_IMG_SIZE": 800,
    "TRAIN_IMG_NUM": 63182,
    "AUGMENT": True,
    "MULTI_SCALE_TRAIN": True,
    "MULTI_TRAIN_RANGE": [23, 28, 1],
    "BATCH_SIZE": 18,
    "IOU_THRESHOLD_LOSS": 0.6,
    "EPOCHS": 51,
    "NUMBER_WORKERS": 8,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 2e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 3,
    "IOU_TYPE": 'GIOU'
}

TEST = {
    "EVAL_TYPE": 'VOC',
    "EVAL_JSON": 'test.json',
    "EVAL_NAME": 'test',
    "NUM_VIS_IMG": 0,
    "TEST_IMG_SIZE": 800,
    "BATCH_SIZE": 4,
    "NUMBER_WORKERS": 8,
    "CONF_THRESH": 0.05,
    "NMS_THRESH": 0.45,
    "IOU_THRESHOLD": 0.5,
    "NMS_METHODS": 'NMS',
    "MULTI_SCALE_TEST": False,
    "MULTI_TEST_RANGE": [736, 864, 96],
    "FLIP_TEST": False
}






