#!/usr/bin/env python
# -*- coding:utf-8 -*-
# author: Ti Bai
# datetime:2019/7/2 15:58

# sys
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import argparse
import importlib
import shutil

# torch
import torch
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

# monai
from monai.config import print_config

# project
from lib.config import cfg, update_config
from lib.utils.utils import create_logger, save_checkpoint
from lib.engine.train_engine import do_train
from lib.engine.validate_engine import do_validate
from lib.engine.test_engine import do_test

import lib.dataset as dataset
import lib.model as model


def parse_args():
    parser = argparse.ArgumentParser(description="Image to image translation")
    parser.add_argument('--cfg', default=r'experiments\AAPMLowDose.yaml', type=str)
    parser.add_argument('output_dir', default=None, type=str, nargs='?')
    parser.add_argument('log_dir', default=None, type=str, nargs='?')
    parser.add_argument('data_root', default=None, type=str, nargs='?')
    parser.add_argument('opts',
                        help="Modify config options using the command-line",
                        default=None,
                        nargs=argparse.REMAINDER)
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    update_config(cfg, args)

    # cudnn related setting
    cudnn.benchmark = cfg.CUDNN.BENCHMARK
    torch.backends.cudnn.deterministic = cfg.CUDNN.DETERMINISTIC
    torch.backends.cudnn.enabled = cfg.CUDNN.ENABLED

    # logger
    logger, final_output_dir, tb_log_dir, key_files_dir = create_logger(cfg, args.cfg)
    logger.info(cfg)
    print_config()

    # move key files into the folder to ensure the reproducibility
    shutil.copy(args.cfg, os.path.join(key_files_dir, os.path.basename(args.cfg)))
    shutil.copy('main.py', os.path.join(key_files_dir, 'main.py'))
    shutil.copytree('./lib', os.path.join(key_files_dir, 'lib'),
                    ignore=shutil.ignore_patterns('*.pyc', 'tmp*', '*__pycache__*', '*bin', '*npy'))

    # data
    train_loader = eval('dataset.' + cfg.DATASET.NAME + '.get_data_provider')(cfg, phase='train')
    val_loader = eval('dataset.' + cfg.DATASET.NAME + '.get_data_provider')(cfg, phase='val')
    test_loader = eval('dataset.' + cfg.DATASET.NAME + '.get_data_provider')(cfg, phase='test')

    # model
    model = eval('model.' + cfg.MODEL.NAME + '.get_model')(cfg, is_train=True)
    model.setup(cfg)

    # visualize function
    visualize_function = None
    if cfg.IS_VISUALIZE:
        try:
            visualize_module = importlib.import_module(r'lib.analyze.visualize_{}'.format(cfg.MODEL.NAME))
        except:
            logger.info('Cannot find visualize function: visualize_{}! Using the default function!'.format(cfg.MODEL.NAME))
            visualize_module = importlib.import_module(r'lib.analyze.visualize_Default')

        visualize_function = getattr(visualize_module, 'visualize')

    # setup the iteration indicator
    writer_dict = {
        'writer': SummaryWriter(log_dir=tb_log_dir),
        'train_global_steps': 0,
        'val_global_steps': 0,
        'test_global_steps': 0
    }

    indicator_dict = {"current_performance": 1e8,
                      "best_performance": 1e8,
                      "is_best": False,
                      "current_iteration": 0,
                      "total_iteration": cfg.TRAIN.TOTAL_ITERATION
                      }

    # auto-resume
    checkpoint_file = cfg.TRAIN.CHECKPOINT
    if cfg.AUTO_RESUME and os.path.exists(checkpoint_file):
        logger.info('=> loading checkpoint {}'.format(checkpoint_file))
        checkpoint = torch.load(checkpoint_file)

        indicator_dict = checkpoint['indicator_dict']
        indicator_dict['total_iteration'] = cfg.TRAIN.TOTAL_ITERATION
        indicator_dict['current_iteration'] = checkpoint['indicator_dict']['current_iteration']

        setup_dict = {'last_iteration': indicator_dict['current_iteration']}
        state_keys = ['generator', 'optimizer_generator']
        for current_key in state_keys:
            if current_key in checkpoint:
                setup_dict[current_key] = checkpoint[current_key]
        model.setup(setup_dict)

        writer_dict['train_global_steps'] = checkpoint['writer_dict_train_global_steps']
        writer_dict['val_global_steps'] = checkpoint['writer_dict_val_global_steps']
        writer_dict['writer'] = SummaryWriter(log_dir=checkpoint['tb_log_dir'])

        logger.info('=> loaded checkpoint {} from iteration {}'.format(checkpoint_file,
                                                                       indicator_dict['current_iteration']))

    logger.info(indicator_dict)

    if True:
        do_train(train_loader,
                 val_loader,
                 model,
                 indicator_dict,
                 cfg,
                 writer_dict,
                 final_output_dir,
                 tb_log_dir,
                 visualize_function)

        do_validate(val_loader, model, cfg, visualize_function, writer_dict, final_output_dir)

    if True:
        output_dictionary = {'indicator_dict': indicator_dict,
                             'writer_dict_train_global_steps': writer_dict['train_global_steps'],
                             'writer_dict_val_global_steps': writer_dict['val_global_steps'],
                             'tb_log_dir': tb_log_dir}

        if hasattr(model, 'generator'):
            output_dictionary['generator'] = model.generator.state_dict()
        else:
            raise ModuleNotFoundError("Not find the generator!")

        save_checkpoint(output_dictionary, indicator_dict, final_output_dir, filename='model_final.pth.tar')

    if True:
        do_test(test_loader, model, cfg, visualize_function, writer_dict, final_output_dir)
    writer_dict['writer'].close()


if __name__ == '__main__':
    main()
    print('Congrats! May the force be with you ...')
