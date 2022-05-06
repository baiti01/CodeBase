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
    end = time.time()
    model.eval()

    for i, current_data in enumerate(test_loader):
        is_success = model.set_dataset(current_data)
        if is_success == -1:
            continue

        model.input.require_grad = False

        with torch.no_grad():
            model.forward()
            model.loss_calculation()

        batch_time.update(time.time() - end)
        end = time.time()

        performance = model.record_information(current_iteration=i,
                                               data_loader_size=len(test_loader),
                                               writer_dict=writer_dict,
                                               phase='val')

        if True:
            visualize(model, i, os.path.join(final_output_dir, "test"), 1 )
