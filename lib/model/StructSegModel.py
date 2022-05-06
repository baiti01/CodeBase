#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 7/21/2020 7:02 PM

# sys
import logging

import torch
from easydict import EasyDict as edict

# monai
from monai.metrics import compute_meandice
from monai.networks.utils import one_hot

# project
from lib.model.base_model import BaseModel
from lib.model.module.networks import define_generator
from lib.utils.utils import AverageMeter

logger = logging.getLogger(__name__)


class StructSegModel(BaseModel):
    def __init__(self,
                 optimizer_option,
                 criterion_option,
                 scheduler_option,
                 cfg=None,
                 is_train=True
                 ):
        super(StructSegModel, self).__init__(is_train=is_train)

        self.generator = define_generator(cfg.MODEL.GENERATOR)
        self._create_optimize_engine(optimizer_option, criterion_option, scheduler_option)
        self.cfg = cfg

    def forward(self):
        self.output = self.generator(self.input)

    def loss_calculation(self):
        unique_label_idx = torch.unique(self.target)
        if len(unique_label_idx) == 1:
            return -1
        self.target = one_hot(self.target, num_classes=self.cfg.MODEL.GENERATOR.OUTPUT_CHANNELS)
        self.class_spatial_mask = torch.zeros_like(self.target)
        self.class_spatial_mask[:, unique_label_idx.long()] = 1
        self.loss = self.criterion_pixel_wise_loss(self.output, self.target, self.class_spatial_mask)
        self.loss = self.loss * (self.cfg.MODEL.GENERATOR.OUTPUT_CHANNELS - 1) / (len(unique_label_idx) - 1)
        return 0

    def optimize_parameters(self):
        self.forward()
        is_success = self.loss_calculation()
        if is_success == -1:
            return -1
        self.optimizer_generator.zero_grad()
        self.loss.backward()
        self.optimizer_generator.step()
        return 0

    def record_information(self, current_iteration=None, data_loader_size=None, batch_time=None, data_time=None,
                           indicator_dict=None, writer_dict=None, phase='train'):
        writer = writer_dict['writer']
        if phase == 'train':
            self.losses_train.update(self.loss.item())
            indicator_dict['current_iteration'] += 1
            global_steps = writer_dict['train_global_steps']
            writer.add_scalar('train_loss', self.loss.item(), global_steps)
            writer_dict['train_global_steps'] = global_steps + 1

            if current_iteration % self.cfg.TRAIN.PRINT_FREQUENCY == 0:
                msg = 'Iteration: [{0}/{1}]\t' \
                      'Time {batch_time.val:.3f}s ({batch_time.avg:.3f}s)\t' \
                      'Data {data_time.val:.3f}s ({data_time.avg:.3f}s)\t' \
                      'LR: {LR:.6f}\t' \
                      'Loss {losses.val:.5f} ({losses.avg:.5f})'.format(
                    current_iteration, data_loader_size,
                    batch_time=batch_time,
                    data_time=data_time,
                    LR=self.schedulers[0].get_last_lr()[0],
                    losses=self.losses_train)
        elif phase == 'val':
            if current_iteration == 0:
                self.losses_val = AverageMeter()
                self.DSC = AverageMeter()
            self.losses_val.update(self.loss.item())
            current_DSC = compute_meandice(self.output > 0, self.target, include_background=False)
            self.DSC.update(current_DSC.detach().cpu().numpy())

            if current_iteration == data_loader_size - 1:
                global_steps = writer_dict['val_global_steps']
                writer.add_scalar('val_loss', self.loss, global_steps)
                writer_dict['val_global_steps'] = global_steps + 1

            if current_iteration % self.cfg.VAL.PRINT_FREQUENCY == 0:
                msg = 'Val: [{0}/{1}]\t' \
                      'Loss {losses.val:.5f} ({losses.avg:.5f})\t' \
                      'DSC {DSCes.val[0]} ({DSCes.avg})'.format(
                    current_iteration, data_loader_size,
                    losses=self.losses_val,
                    DSCes=self.DSC)
        else:
            raise ValueError('Unknown operation in information recording!')
        logger.info(msg)
        return self.losses_val.avg


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

    model = StructSegModel(optimizer_option=optimizer_option,
                           criterion_option=criterion_option,
                           scheduler_option=scheduler_option,
                           cfg=cfg,
                           is_train=is_train)

    return model
