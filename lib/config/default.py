#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai
# datetime:2019/7/9 15:53

# sys
from yacs.config import CfgNode as CN
import yaml

_C = CN()

# miscellaneous
_C.AUTO_RESUME = True
_C.WORKERS = 0
_C.PIN_MEMORY = True
_C.OUTPUT_DIR = 'OUTPUT'
_C.IS_VISUALIZE = True

# CUDNN related params
_C.CUDNN = CN()
_C.CUDNN.BENCHMARK = True
_C.CUDNN.DETERMINISTIC = False
_C.CUDNN.ENABLED = True

# common parameters for network
_C.MODEL = CN(new_allowed=True)
_C.MODEL.NAME = ''

# generator
_C.MODEL.GENERATOR = CN(new_allowed=True)
_C.MODEL.GENERATOR.PRETRAINED = ''

# discriminator
_C.MODEL.DISCRIMINATOR = CN(new_allowed=True)
_C.MODEL.DISCRIMINATOR.PRETRAINED = ''

# dataset
_C.DATASET = CN(new_allowed=True)
_C.DATASET.NAME = ''
_C.DATASET.ROOT = ''
_C.DATASET.TRAIN_LIST = 'train.txt'
_C.DATASET.VAL_LIST = 'val.txt'
_C.DATASET.TEST_LIST = 'test.txt'


# loss
_C.CRITERION = CN()
_C.CRITERION.PIXEL_WISE_LOSS_TYPE = 'mse'
_C.CRITERION.DISCRIMINATOR_LOSS_TYPE = 'ce'
_C.CRITERION.EXTRA = CN(new_allowed=True)

# train
_C.TRAIN = CN()
_C.TRAIN.TOTAL_ITERATION = 1000000.0
_C.TRAIN.CHECKPOINT = ''

# lr
_C.TRAIN.LR_POLICY = 'MultiStepLR'
_C.TRAIN.LR_STEP = (50000, 75000)
_C.TRAIN.LR_FACTOR = 0.1
_C.TRAIN.OPTIMIZER = 'adam'

# adam
_C.TRAIN.GAMMA1 = 0.5
_C.TRAIN.GAMMA2 = 0.999

# sgd
_C.TRAIN.MOMENTUM = 0.9
_C.TRAIN.WD = 0.0
_C.TRAIN.NESTEROV = True

# train miscellaneous
_C.TRAIN.BATCHSIZE_PER_GPU = 1
_C.TRAIN.SHUFFLE = True
_C.TRAIN.PRINT_FREQUENCY = 1.0
_C.TRAIN.DISPLAY_FREQUENCY = 100
_C.TRAIN.SAVE_EPOCH_FREQUENCY = 1

_C.TRAIN.GENERATOR = CN()
_C.TRAIN.GENERATOR.LR = 0.0003

_C.TRAIN.DISCRIMINATOR = CN()
_C.TRAIN.DISCRIMINATOR.LR = 0.0003

# val
_C.VAL = CN()
_C.VAL.BATCHSIZE_PER_GPU = 1
_C.VAL.SHUFFLE = False
_C.VAL.MODEL_FILE = ''
_C.VAL.EVALUATION_FREQUENCY = 1
_C.VAL.PRINT_FREQUENCY = 1.0

# test
_C.TEST = CN()
_C.TEST.BATCHSIZE_PER_GPU = 1
_C.TEST.IS_VOLUME_VISUALIZER = True
_C.TEST.SHUFFLE = False
_C.TEST.MODEL_FILE = ''
_C.TEST.PRINT_FREQUENCY = 1.0


def update_config(cfg, args):
    cfg.defrost()

    with open(args.cfg) as f:
        config_params = yaml.load(f)

    cfg.merge_from_file(args.cfg)
    cfg.merge_from_list(args.opts)

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.log_dir:
        cfg.LOG_DIR = args.log_dir

    if args.data_root:
        cfg.DATASET.ROOT = args.data_root

    if hasattr(args, 'test_model'):
        cfg.TEST.MODEL_FILE = args.test_model

    cfg.freeze()


if __name__ == '__main__':
    print(_C)
