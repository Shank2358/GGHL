DATA_PATH = "/home/hzc/DroneVehicle/"
PROJECT_PATH = "./"

# DATA = {"CLASSES": ['plane',
#                     'baseball-diamond',
#                     'bridge',
#                     'ground-track-field',
#                     'small-vehicle',
#                     'large-vehicle',
#                     'ship',
#                     'tennis-court',
#                     'basketball-court',
#                     'storage-tank', 'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter'],#,'container-crane', 'airport', 'helipad','container-crane', 'airport', 'helipad'
#         "NUM": 15}

DATA = {"CLASSES": ['car', 'truck', 'bus', 'van', 'freight car'],
        "NUM": 5}

DATASET_NAME = "trainIR" #"train_DOTAx" 
MODEL = {"ANCHORS":[[(2.80340246, 2.87380792), (4.23121697, 6.44043634), (7.38428433, 3.82613533)],
        [(4.2460819, 4.349495965), (4.42917327, 10.59395029), (8.24772929, 6.224761455)],
        [(6.02687863, 5.92446062), (7.178407523, 10.86361071), (15.30253702, 12.62863728)]] ,
         "STRIDES":[8, 16, 32], "SCALES_PER_LAYER": 3}

MAX_LABEL = 1000
SHOW_HEATMAP = False
SCALE_FACTOR = 2.0
SCALE_FACTOR_A = 2.0

TRAIN = {
    "Transformer_SIZE": 896,
    "EVAL_TYPE": 'VOC',
    "TRAIN_IMG_SIZE": 800,
    "TRAIN_IMG_NUM": 79780,
    "AUGMENT": True,
    "MULTI_SCALE_TRAIN": True,
    "MULTI_TRAIN_RANGE": [23, 28, 1],
    "BATCH_SIZE": 36,
    "IOU_THRESHOLD_LOSS": 0.5,
    "EPOCHS": 100,
    "NUMBER_WORKERS": 8,
    "MOMENTUM": 0.9,
    "WEIGHT_DECAY": 0.0005,
    "LR_INIT": 2e-4,
    "LR_END": 1e-6,
    "WARMUP_EPOCHS": 5,
    "IOU_TYPE": 'GIOU'
}

TEST = {
    "EVAL_TYPE": 'VOC',
    "EVAL_JSON": 'test.json',
    "EVAL_NAME": 'test',
    "NUM_VIS_IMG": 0,
    "TEST_IMG_SIZE": 896,
    "BATCH_SIZE": 4,
    "NUMBER_WORKERS": 16,
    "CONF_THRESH": 0.06,
    "NMS_THRESH": 0.4,
    "IOU_THRESHOLD": 0.5,
    "NMS_METHODS": 'NMS',
    "MULTI_SCALE_TEST": False,
    "MULTI_TEST_RANGE": [832, 992, 32],
    "FLIP_TEST": False
}






