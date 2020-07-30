#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 7/20/2020 11:10 PM

from yacs.config import CfgNode as CN

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
