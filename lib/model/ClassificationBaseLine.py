#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 7/21/2020 7:02 PM

# sys
import logging
from easydict import EasyDict as edict

# torch
import torch

# project
from lib.model.base_model import BaseModel
import torchvision.models as models

logger = logging.getLogger(__name__)

model_dict = {'mobilenet_v2': models.mobilenet_v2,
              'resnet50': models.resnet50,
              'densenet121': models.densenet121,
              'resnext50_32x4d': models.resnext50_32x4d,
              'inception_v3': models.inception_v3,
              'resnext101_32x8d': models.resnext101_32x8d}


class ClassificationBaseLine(BaseModel):
    def __init__(self,
                 optimizer_option,
                 criterion_option,
                 scheduler_option,
                 cfg=None,
                 is_train=True
                 ):
        super(ClassificationBaseLine, self).__init__(is_train=is_train)
        self.discriminator = model_dict[cfg.MODEL.DISCRIMINATOR.NAME](num_classes=cfg.MODEL.NUM_CLASSES * cfg.DATASET.DISEASE_NUMBER)
        self._create_optimize_engine(optimizer_option, criterion_option, scheduler_option)
        self.discriminator_loss = torch.nn.CrossEntropyLoss(ignore_index=100)
        if torch.cuda.is_available():
            self.discriminator_loss = self.discriminator_loss.cuda()
            self.discriminator = self.discriminator.cuda()
        self.cfg = cfg

    def forward(self):
        self.output = self.discriminator(self.input)
        self.output = self.output.view(self.output.shape[0], -1, self.cfg.DATASET.DISEASE_NUMBER)

    def loss_calculation(self):
        self.loss = self.discriminator_loss(self.output, self.target)

    def optimize_parameters(self):
        self.forward()
        self.loss_calculation()
        self.optimizer_discriminator.zero_grad()
        self.loss.backward()
        self.optimizer_discriminator.step()


def get_model(cfg, is_train=True):
    optimizer_option = edict({'optimizer': cfg.TRAIN.OPTIMIZER,
                              'discriminator_lr': cfg.TRAIN.DISCRIMINATOR.LR,
                              'beta1': cfg.TRAIN.GAMMA1,
                              'beta2': cfg.TRAIN.GAMMA2})

    criterion_option = edict({
        'discriminator_loss_type': cfg.CRITERION.DISCRIMINATOR_LOSS_TYPE,
    })

    scheduler_option = edict({'niter_decay': int(cfg.TRAIN.TOTAL_ITERATION),
                              'lr_policy': cfg.TRAIN.LR_POLICY,
                              'lr_decay_iters': cfg.TRAIN.LR_STEP,
                              'last_iteration': -1})

    model = ClassificationBaseLine(optimizer_option=optimizer_option,
                                   criterion_option=criterion_option,
                                   scheduler_option=scheduler_option,
                                   cfg=cfg,
                                   is_train=is_train)

    return model
