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

logger = logging.getLogger(__name__)


class BaseModel(nn.Module):
    def __init__(self, is_train=True):
        super(BaseModel, self).__init__()
        self.is_train = is_train

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

            if criterion_option.pixel_wise_loss_type == 'mse':
                self.criterion_pixel_wise_loss = torch.nn.MSELoss()
            elif criterion_option.pixel_wise_loss_type == 'L1':
                self.criterion_pixel_wise_loss = torch.nn.L1Loss()
            else:
                raise "Loss type {} is not implemented yet!".format(criterion_option.pixle_wise_loss_type)

            if torch.cuda.is_available():
                self.criterion_pixel_wise_loss = self.criterion_pixel_wise_loss.cuda()

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
