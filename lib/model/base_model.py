#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 7/21/2020 10:03 AM

# sys
import logging

# torch
import torch
import torch.nn as nn

# project
from lib.model.utils.utils import get_scheduler
from lib.utils.utils import AverageMeter

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, is_train=True):
        super(BaseModel, self).__init__()
        self.is_train = is_train
        self.losses_train = AverageMeter()
        self.losses_val = AverageMeter()

    def _create_optimize_engine(self, optimizer_option, criterion_option, scheduler_option):
        self.optimizers = []
        if self.is_train:
            if optimizer_option.optimizer == 'adam':
                if hasattr(self, 'generator'):
                    self.optimizer_generator = torch.optim.Adam(self.generator.parameters(),
                                                                lr=optimizer_option.generator_lr,
                                                                betas=(optimizer_option.beta1,
                                                                       optimizer_option.beta2))
                    self.optimizers.append(self.optimizer_generator)
                if hasattr(self, 'discriminator'):
                    self.optimizer_discriminator = torch.optim.Adam(self.discriminator.parameters(),
                                                                    lr=optimizer_option.discriminator_lr,
                                                                    betas=(optimizer_option.beta1,
                                                                           optimizer_option.beta2))
                    self.optimizers.append(self.optimizer_discriminator)
            elif optimizer_option.optimizer == 'sgd':
                if hasattr(self, 'generator'):
                    self.optimizer_generator = torch.optim.SGD(self.generator.parameters(),
                                                               lr=optimizer_option.generator_lr,
                                                               momentum=optimizer_option.momentum,
                                                               weight_decay=optimizer_option.weight_decay,
                                                               nesterov=optimizer_option.nesterov)
                    self.optimizers.append(self.optimizer_generator)
                if hasattr(self, 'discriminator'):
                    self.optimizer_generator = torch.optim.SGD(self.discriminator.parameters(),
                                                               lr=optimizer_option.discriminator_lr,
                                                               momentum=optimizer_option.momentum,
                                                               weight_decay=optimizer_option.weight_decay,
                                                               nesterov=optimizer_option.nesterov)
                    self.optimizers.append(self.optimizer_discriminator)
            else:
                raise NotImplementedError(
                    'Optimizer type {} is not implemented yet!'.format(optimizer_option.optimizer))

            if hasattr(criterion_option, 'pixel_wise_loss_type'):
                if criterion_option.pixel_wise_loss_type == 'mse':
                    self.criterion_pixel_wise_loss = torch.nn.MSELoss()
                elif criterion_option.pixel_wise_loss_type == 'L1':
                    self.criterion_pixel_wise_loss = torch.nn.L1Loss()
                elif criterion_option.pixel_wise_loss_type == 'ClassSpatialMaskedDiceLoss':
                    from lib.model.loss.ClassSpatialMaskedLoss import ClassSpatialMaskedDiceLoss
                    self.criterion_pixel_wise_loss = ClassSpatialMaskedDiceLoss(include_background=False, sigmoid=False, to_onehot_y=False)
                elif criterion_option.pixel_wise_loss_type == 'DiceLoss':
                    from monai.losses import DiceLoss
                    self.criterion_pixel_wise_loss = DiceLoss(include_background=False, softmax=True, to_onehot_y=True)
                elif criterion_option.pixel_wise_loss_type == 'DiceCELoss':
                    from monai.losses import DiceCELoss
                    self.criterion_pixel_wise_loss = DiceCELoss(include_background=False, to_onehot_y=True, softmax=True)
                elif criterion_option.pixel_wise_loss_type == 'DiceFocalLoss':
                    from monai.losses import DiceFocalLoss
                    self.criterion_pixel_wise_loss = DiceFocalLoss(include_background=False, to_onehot_y=True, softmax=True)
                else:
                    raise "Loss type {} is not implemented yet!".format(criterion_option.pixle_wise_loss_type)

                if torch.cuda.is_available():
                    self.criterion_pixel_wise_loss = self.criterion_pixel_wise_loss.cuda()

            if hasattr(criterion_option, 'discriminator_loss_type'):
                if criterion_option.discriminator_loss_type == 'ce':
                    self.discriminator_loss = torch.nn.CrossEntropyLoss()
                elif criterion_option.classification_loss_type == 'bce':
                    self.discriminator_loss = torch.nn.BCEWithLogitsLoss()
                else:
                    raise "Loss type {} is not implemented yet!".format(criterion_option.discriminator_loss_type)
                if torch.cuda.is_available():
                    self.discriminator_loss = self.discriminator_loss.cuda()

            self.scheduler_option = scheduler_option

    def setup(self, options=None):
        """ load and print networks; create schedulers
        """
        self.last_iteration = -1
        self.print_network()

        if options is not None:
            if 'generator' in options:
                self.generator.load_state_dict(options['generator'])
                logger.info('Finish generator loading!')
            else:
                logger.info('Cannot find pretrained generator! Train from scratch!')

            if 'optimizer_generator' in options:
                self.optimizer_generator.load_state_dict(options['optimizer_generator'])
                logger.info('Finish optimizer_generator loading!')
            else:
                logger.info('Cannot find the state of the optimizer_generator! Record from beginning!')

            if 'discriminator' in options:
                self.generator.load_state_dict(options['discriminator'])
                logger.info('Finish discriminator loading!')
            else:
                logger.info('Cannot find pretrained discriminator! Train from scratch!')

            if 'optimizer_discriminator' in options:
                self.optimizer_generator.load_state_dict(options['optimizer_discriminator'])
                logger.info('Finish optimizer_discriminator loading!')
            else:
                logger.info('Cannot find the state of the optimizer_discriminator! Record from beginning!')

            if 'last_iteration' in options:
                self.last_iteration = options['last_iteration']
                logger.info('Train from iteration {} ...'.format(self.last_iteration))
                self.scheduler_option['last_iteration'] = self.last_iteration
            else:
                logger.info('Cannot find last iteration index! Train from iteration 0!')

        if self.is_train:
            self.schedulers = [get_scheduler(current_optimizer, self.scheduler_option)
                               for current_optimizer in self.optimizers]

    def print_network(self):
        if hasattr(self, 'generator'):
            logger.info("=== generator ===")
            logger.info(self.generator)

        if hasattr(self, 'discriminator'):
            logger.info("=== discriminator ===")
            logger.info(self.discriminator)

    def set_dataset(self, input):
        self.input = input['input']
        self.target = input['target']
        self.data_path = input['path']
        if torch.cuda.is_available():
            self.input = self.input.cuda()
            self.target = self.target.cuda()

        self.target.requires_grad = False

    def forward(self):
        pass

    def loss_calculation(self):
        pass

    def optimize_parameters(self):
        pass

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
            self.losses_val.update(self.loss.item())

            if current_iteration == data_loader_size - 1:
                global_steps = writer_dict['val_global_steps']
                writer.add_scalar('val_loss', self.loss, global_steps)
                writer_dict['val_global_steps'] = global_steps + 1

            if current_iteration % self.cfg.VAL.PRINT_FREQUENCY == 0:
                msg = 'Val: [{0}/{1}]\t' \
                      'Loss {losses.val:.5f} ({losses.avg:.5f})'.format(
                    current_iteration, data_loader_size,
                    losses=self.losses_val)
        else:
            raise ValueError('Unknown operation in information recording!')
        logger.info(msg)
        return self.losses_val.avg

