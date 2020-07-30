#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 8/12/2019 3:43 PM

# sys
import time
import logging
import os
import random

# torch
import torch

# project
from lib.utils.utils import AverageMeter

logger = logging.getLogger(__name__)


def do_validate(val_loader,
                model,
                cfg,
                visualize,
                writer_dict,
                final_output_dir):
    batch_time = AverageMeter()
    end = time.time()
    model.eval()

    selected_visualized_data = random.randint(0, len(val_loader) - 1)
    for i, current_data in enumerate(val_loader):
        model.set_dataset(current_data)

        with torch.no_grad():
            model.forward()
            model.loss_calculation()

        batch_time.update(time.time() - end)
        end = time.time()

        performance = model.record_information(current_iteration=i,
                                               data_loader_size=len(val_loader),
                                               writer_dict=writer_dict,
                                               phase='val')
        if i == selected_visualized_data:
            visualize(model,
                      writer_dict['val_global_steps'],
                      os.path.join(final_output_dir, "val"),
                      1
                      )

    return performance
