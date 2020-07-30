#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 1/17/2020 11:30 AM

# torch
import torch.utils.data as data


class BaseDataset(data.Dataset):
    def __init__(self, before_min=0, before_max=2000, after_min=0, after_max=2000):
        super(BaseDataset, self).__init__()
        self.before_min = before_min
        self.before_max = before_max

        self.after_min = after_min
        self.after_max = after_max

    @staticmethod
    def __calculate_anchor_location__(origin_size, target_size, target_shift_size):
        if target_shift_size * 2 > origin_size - target_size:
            target_shift_size = (origin_size - target_size) / 2

        start = (origin_size / 2 - target_shift_size) - target_size / 2
        end = (origin_size / 2 + target_shift_size) - target_size / 2
        return start, end

    def _normalization(self, x):
        x = (x - self.before_min) / (self.before_max - self.before_min)
        x = x * (self.after_max - self.after_min) + self.after_min
        return x
