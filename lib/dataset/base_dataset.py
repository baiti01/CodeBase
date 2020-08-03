#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 1/17/2020 11:30 AM

# sys
import cv2
import numpy as np

# torch
import torch.utils.data as data
import torchvision.transforms as tfs


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

    def _border_pad(self, image, border_type='zero', long_side=512, pixel_mean=0):
        h, w, c = image.shape
        constant_value = 0.0 if border_type == 'zero' else pixel_mean
        if border_type == 'zero' or border_type == 'pixel_mean':
            image = np.pad(image, ((0, long_side - h), (0, long_side - w), (0, 0)),
                           mode='constant', constant_values=constant_value)
        else:
            image = np.pad(image, ((0, long_side - h), (0, long_side - w), (0, 0)), mode=border_type)
        return image

    def _fix_ratio(self, image, border_type='zero', long_side=512):
        h, w, c = image.shape

        if h >= w:
            ratio = h * 1.0 / w
            h_ = long_side
            w_ = round(h_ / ratio)
        else:
            ratio = w * 1.0 / h
            w_ = long_side
            h_ = round(w_ / ratio)

        image = cv2.resize(image, dsize=(w_, h_), interpolation=cv2.INTER_LINEAR)
        image = self._border_pad(image, border_type=border_type, long_side=long_side)
        return image

    def Common(self, image):
        image = cv2.equalizeHist(image)
        image = cv2.GaussianBlur(image, (3, 3), 0)
        image = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB).astype(np.float32)
        return image

    def Aug(self, image):
        img_aug = tfs.Compose([tfs.RandomAffine(degrees=(-15, 15),
                                                translate=(0.05, 0.05),
                                                scale=(0.95, 1.05),
                                                fillcolor=128)])
        image = img_aug(image)
        return image

    def GetTransforms(self, image, target=None, type='Aug'):
        # taget is not support now
        if target is not None:
            raise Exception('Target is not support now ! ')

        # get type
        if type.strip() == 'Common':
            image = self.Common(image)
            return image
        elif type.strip() == 'None':
            return image
        elif type.strip() == 'Aug':
            image = self.Aug(image)
            return image
        else:
            raise Exception(
                'Unknown transforms_type : '.format(type))
