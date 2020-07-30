#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai
# datetime:2019/7/3 14:32

# sys
import os
import time
import logging
from pathlib import Path
import numpy as np

# torch
import torch


def epoch_iteration_calculation(total_iterations, batch_size, length_dataset, drop_last=False):
    iteration_per_epoch = length_dataset/batch_size
    iteration_per_epoch = np.floor(iteration_per_epoch) if drop_last else np.ceil(iteration_per_epoch)
    total_epochs = np.ceil(total_iterations / iteration_per_epoch)
    return total_epochs, iteration_per_epoch


def save_checkpoint(states, indicator_dict, output_dir,
                    filename='checkpoint.pth.tar'):
    torch.save(states, os.path.join(output_dir, filename))

    with open(os.path.join(output_dir, 'model_info.log'), 'a') as f:
        current_line = 'checkpoint: \t {} \t {:.8f} \t'.format(indicator_dict['current_iteration'],
                                                              indicator_dict['current_performance'])
        is_best_info = 'is_best' if indicator_dict['is_best'] else 'no_best'
        current_line = current_line + is_best_info + '\n'
        f.write(current_line)

    if indicator_dict['is_best']:
        torch.save(states,
                   os.path.join(output_dir, 'model_best.pth.tar'))


def create_logger(cfg, cfg_name, phase='train'):
    root_output_dir = Path(cfg.OUTPUT_DIR)
    # set up logger
    if not root_output_dir.exists():
        print('=> creating {}'.format(root_output_dir))
        root_output_dir.mkdir()

    dataset = cfg.DATASET.NAME
    cfg_name = os.path.basename(cfg_name).split('.')[0]

    time_str = time.strftime('%Y-%m-%d-%H-%M-%S')
    final_output_dir = root_output_dir / dataset / cfg_name / time_str
    print('=> creating {}'.format(final_output_dir))
    final_output_dir.mkdir(parents=True, exist_ok=True)
    
    if phase == 'train':
        final_output_dir_train = final_output_dir / "train"
        final_output_dir_val = final_output_dir / "val"
        final_output_dir_volume = final_output_dir / "volume"

        final_output_dir_train.mkdir(parents=True, exist_ok=True)
        final_output_dir_val.mkdir(parents=True, exist_ok=True)
        final_output_dir_volume.mkdir(parents=True, exist_ok=True)
    elif phase == 'test':
        final_output_dir_test = final_output_dir / "test"
        final_output_dir_volume = final_output_dir / "volume"

        final_output_dir_test.mkdir(parents=True, exist_ok=True)
        final_output_dir_volume.mkdir(parents=True, exist_ok=True)

    log_file = '{}_{}_{}.log'.format(cfg_name, time_str, phase)
    final_log_file = final_output_dir / log_file
    head = '%(asctime)-15s %(message)s'
    logging.basicConfig(filename=str(final_log_file),
                        format=head)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    console = logging.StreamHandler()
    logging.getLogger('').addHandler(console)

    tensorboard_log_dir = Path(cfg.LOG_DIR) / dataset / cfg_name / time_str

    print('=> creating {}'.format(tensorboard_log_dir))
    tensorboard_log_dir.mkdir(parents=True, exist_ok=True)

    return logger, str(final_output_dir), str(tensorboard_log_dir)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

