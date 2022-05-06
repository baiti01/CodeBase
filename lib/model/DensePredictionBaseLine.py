#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 7/21/2020 7:02 PM

# sys
import logging
from easydict import EasyDict as edict

# project
from lib.model.base_model import BaseModel
from lib.model.module.networks import define_generator

logger = logging.getLogger(__name__)


class DensePredictionBaseLine(BaseModel):
    def __init__(self,
                 optimizer_option,
                 criterion_option,
                 scheduler_option,
                 cfg=None,
                 is_train=True
                 ):
        super(DensePredictionBaseLine, self).__init__(is_train=is_train)

        self.visualized_images = ['input', 'output', 'target', 'diff']
        self.generator = define_generator(cfg.MODEL.GENERATOR)
        self._create_optimize_engine(optimizer_option, criterion_option, scheduler_option)
        self.cfg = cfg

    def forward(self):
        self.output = self.generator(self.input)

    def loss_calculation(self):
        self.loss = self.criterion_pixel_wise_loss(self.output, self.target)

    def optimize_parameters(self):
        self.forward()
        self.loss_calculation()
        self.optimizer_generator.zero_grad()
        self.loss.backward()
        self.optimizer_generator.step()


def get_model(cfg, is_train=True):
    optimizer_option = edict({'optimizer': cfg.TRAIN.OPTIMIZER,
                              'generator_lr': cfg.TRAIN.GENERATOR.LR,
                              'beta1': cfg.TRAIN.GAMMA1,
                              'beta2': cfg.TRAIN.GAMMA2})

    criterion_option = edict({
        'pixel_wise_loss_type': cfg.CRITERION.PIXEL_WISE_LOSS_TYPE,
    })

    scheduler_option = edict({'niter_decay': int(cfg.TRAIN.TOTAL_ITERATION),
                              'lr_policy': cfg.TRAIN.LR_POLICY,
                              'lr_decay_iters': cfg.TRAIN.LR_STEP,
                              'last_iteration': -1})

    model = DensePredictionBaseLine(optimizer_option=optimizer_option,
                                    criterion_option=criterion_option,
                                    scheduler_option=scheduler_option,
                                    cfg=cfg,
                                    is_train=is_train)
    return model
