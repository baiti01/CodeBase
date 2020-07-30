#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MIAI Lab | UT Southwestern Medical Center
# DATETIME: 12/18/2019 3:20 PM

# torch
import torch
import torch.nn as nn
from torch.optim import lr_scheduler


def get_scheduler(optimizer, option):
    """Return a learning rate scheduler

    Parameters:
        optimizer          -- the optimizer of the network
        option (option class) -- stores all the experiment flags; needs to be a subclass of BaseOptions．　
                              opt.lr_policy is the name of learning rate policy: linear | step | plateau | cosine

    For 'linear', we keep the same learning rate for the first <opt.niter> epochs
    and linearly decay the rate to zero over the next <opt.niter_decay> epochs.
    For other schedulers (step, plateau, and cosine), we use the default PyTorch schedulers.
    See https://pytorch.org/docs/stable/optim.html for more details.
    """
    if option.lr_policy == 'linear':
        def lambda_rule(epoch):
            lr_l = 1.0 - max(0, epoch + option.epoch_count - option.niter) / float(option.niter_decay + 1)
            return lr_l
        scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lambda_rule, last_epoch=option.last_iteration)
    elif option.lr_policy == 'step':
        scheduler = lr_scheduler.StepLR(optimizer, step_size=option.lr_decay_iters, gamma=0.1,
                                        last_epoch=option.last_iteration)
    elif option.lr_policy == 'MultiStepLR':
        scheduler = lr_scheduler.MultiStepLR(optimizer, milestones=option.lr_decay_iters, gamma=0.1,
                                             last_epoch=option.last_iteration)
    elif option.lr_policy == 'plateau':
        scheduler = lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.2, threshold=0.01, patience=5)
    elif option.lr_policy == 'cosine':
        scheduler = lr_scheduler.CosineAnnealingLR(optimizer, T_max=option.niter, eta_min=0,
                                                   last_epoch=option.last_iteration)
    else:
        return NotImplementedError('learning rate policy [%s] is not implemented', option.lr_policy)
    return scheduler


def weights_initialization(net, init_type='normal', init_gain=0.02):
    def init_func(m):  # define the initialization function
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                nn.init.normal_(m.weight.data, 0.0, init_gain)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight.data, gain=init_gain)
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight.data, gain=init_gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias.data, 0.0)
        elif classname.find(
                'BatchNorm') != -1:  # BatchNorm Layer's weight is not a matrix; only normal distribution applies.
            nn.init.normal_(m.weight.data, 1.0, init_gain)
            nn.init.constant_(m.bias.data, 0.0)

    #print('initialize network with {}'.format(init_type))
    net.apply(init_func)  # apply the initialization function <init_func>


def network_initialization(net, init_type='normal', init_gain=0.02):
    if torch.cuda.is_available():
        net = torch.nn.DataParallel(net.cuda())
    weights_initialization(net, init_type, init_gain=init_gain)
    return net


if __name__ == '__main__':
    print('Congrats! May the force be with you ...')
