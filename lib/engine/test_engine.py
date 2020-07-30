#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 8/12/2019 3:43 PM
# FILE: train_engine.py

# sys
import time
import logging
import os

# torch
import torch

# project
from lib.utils.utils import AverageMeter

logger = logging.getLogger(__name__)


def do_test(test_loader,
            model,
            cfg,
            visualize,
            writer_dict,
            final_output_dir):
    model.eval()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    SSIM = AverageMeter()
    RMSE = AverageMeter()
    end = time.time()

    writer = writer_dict['writer']

    is_volume_visualizer = cfg.TEST.IS_VOLUME_VISUALIZER
    if is_volume_visualizer:
        from lib.analyze.utilis import VolumeVisualization
        from lib.analyze.utilis import visualize as volume_visualizer_function
        volume_visualizer = VolumeVisualization(data_range=[cfg.DATASET.NORMALIZATION.AFTER_MIN,
                                                            cfg.DATASET.NORMALIZATION.AFTER_MAX],
                                                before_min=cfg.DATASET.NORMALIZATION.BEFORE_MIN,
                                                before_max=cfg.DATASET.NORMALIZATION.BEFORE_MAX,
                                                after_min=cfg.DATASET.NORMALIZATION.AFTER_MIN,
                                                after_max=cfg.DATASET.NORMALIZATION.AFTER_MAX,
                                                threshold=200
                                                )

    for i, current_data in enumerate(test_loader):
        data_time.update(time.time() - end)

        model.set_dataset(current_data)
        with torch.no_grad():
            model.forward()
            #model.output = model.output[0]
            model.target = model.target[0]

        current_loss = model.criterion_pixel_wise_loss(model.output, model.target)
        losses.update(current_loss.item())

        if is_volume_visualizer:
            current_rmse, current_ssim = volume_visualizer.update(current_data, model)
            RMSE.update(current_rmse)
            SSIM.update(current_ssim)

        batch_time.update(time.time() - end)
        end = time.time()

        if i % cfg.VAL.PRINT_FREQUENCY == 0:
            msg = 'Test: [{0}/{1}]\t' \
                  'Loss {losses.val:.5f} ({losses.avg:.5f})\t' \
                  'RMSE {RMSE.val:.5f}({RMSE.avg:.5f})\t' \
                  'SSIM {SSIM.val:.5f}({SSIM.avg:.5f})'.format(
                i, len(test_loader),
                losses=losses,
                RMSE=RMSE,
                SSIM=SSIM)

            logger.info(msg)

        model.output = [model.output]
        model.target = [model.target]
        visualize(model, writer_dict['test_global_steps'],
                  os.path.join(final_output_dir, "test"),
                  1,
                  cfg.DATASET.NORMALIZATION.BEFORE_MIN,
                  cfg.DATASET.NORMALIZATION.BEFORE_MAX,
                  cfg.DATASET.NORMALIZATION.AFTER_MIN,
                  cfg.DATASET.NORMALIZATION.AFTER_MAX)
        writer.add_scalar('test_loss', losses.val, writer_dict['test_global_steps'])
        writer.add_scalar('RMSE', RMSE.val, writer_dict['test_global_steps'])
        writer.add_scalar('SSIM', SSIM.val, writer_dict['test_global_steps'])
        writer_dict['test_global_steps'] += 1

    # log the slice-wise ssim and rmse
    for current_data_path, current_rmse, current_ssim in zip(volume_visualizer.data_path,
                                                             volume_visualizer.rmse,
                                                             volume_visualizer.ssim):
        logger.info('{}\tSSIM:{}\tRMSE:{}'.format(os.path.basename(current_data_path), current_rmse, current_ssim))
    logger.info('* 2D Test: \t Average RMSE {}\t SSIM {}\n'.format(RMSE.avg, SSIM.avg))

    if is_volume_visualizer:
        rmse, ssim = volume_visualizer_function(volume_visualizer,
                                                model,
                                                writer_dict['test_global_steps'],
                                                os.path.join(final_output_dir, "volume"),
                                                1)
        msg = '* 3D Test: \t RMSE {}\t SSIM {}'.format(rmse, ssim)
        logger.info(msg)

    return losses.val
