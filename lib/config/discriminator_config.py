#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 7/20/2020 11:10 PM

# sys
from yacs.config import CfgNode as CN

# torch
import torchvision.models as models

discriminator_dict = {'mobilenet_v2': models.mobilenet_v2,
                      'resnet50': models.resnet50,
                      'densenet121': models.densenet121,
                      'resnext50_32x4d': models.resnext50_32x4d,
                      'inception_v3': models.inception_v3,
                      'resnext101_32x8d': models.resnext101_32x8d}

BASIC = CN()
BASIC.NAME = 'basic'
BASIC.INPUT_CHANNELS = 1
BASIC.NUM_FILTERS = 64
BASIC.NUM_LAYERS = 64
BASIC.NORMALIZATION_TYPE = 'batch'
BASIC.INIT_TYPE = 'normal'
BASIC.INIT_GAIN = 0.02

DISCRIMINATOR_CONFIGS = {
    'BASIC': BASIC
}
