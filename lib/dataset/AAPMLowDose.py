#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MIAI Lab | UT Southwestern Medical Center
# DATETIME: 12/6/2019 3:12 PM

# sys
import os
import numpy as np
import random

# torch
import torch
import torch.utils.data as data

# project
from lib.dataset.base_dataset import BaseDataset


class AAPMLowDose(BaseDataset):
    def __init__(self,
                 crop_size=(512, 512),
                 shift_size=(192, 192),
                 data_root=r'./data',
                 data_list=r'train.list',
                 is_train=True,
                 customized_dataset_size=0
                 ):
        super(AAPMLowDose, self).__init__(before_min=0,
                                          before_max=2000,
                                          after_min=0,
                                          after_max=1)

        if shift_size:
            if isinstance(shift_size, int):
                shift_size = (shift_size, shift_size, shift_size)
            assert isinstance(shift_size, tuple)
        else:
            shift_size = 'maximum'

        self.shift_size = shift_size
        self.crop_size = crop_size
        self.is_train = is_train
        self.data_root = data_root

        self.low_dose_path = []
        self.normal_dose_path = []

        with open(os.path.join(data_root, data_list)) as f:
            for current_line in f.readlines():
                current_low_dose_path, current_normal_dose_path = current_line.strip().split('\t')
                self.low_dose_path.append(os.path.join(data_root, current_low_dose_path))
                self.normal_dose_path.append(os.path.join(data_root, current_normal_dose_path))

        self.real_dataset_size = len(self.low_dose_path)
        self.dataset_size = customized_dataset_size if customized_dataset_size else self.real_dataset_size

    def __getitem__(self, item):
        item = item % self.real_dataset_size
        low_dose_ct_path = self.low_dose_path[item]
        low_dose_ct = np.fromfile(low_dose_ct_path, dtype=np.uint16).reshape(512, 512).astype(np.float32)

        normal_dose_ct_path = self.normal_dose_path[item]
        normal_dose_ct = np.fromfile(normal_dose_ct_path, dtype=np.uint16).reshape(512, 512).astype(np.float32)

        if self.is_train:
            low_dose_ct = np.pad(low_dose_ct, pad_width=16, mode='constant', constant_values=0.0)
            normal_dose_ct = np.pad(normal_dose_ct, pad_width=16, mode='constant', constant_values=0.0)

            # calculate the slice size
            noisy_input_x, noisy_input_y = low_dose_ct.shape
            dim_x, dim_y = self.crop_size
            slice_x, slice_y = dim_x, dim_y

            # calculate the anchor location
            if self.shift_size is 'maximum':
                x_start, x_end = 0, noisy_input_x - slice_x
                y_start, y_end = 0, noisy_input_y - slice_y
            else:
                x_start, x_end = self.__calculate_anchor_location__(noisy_input_x, dim_x, self.shift_size[0])
                y_start, y_end = self.__calculate_anchor_location__(noisy_input_y, dim_y, self.shift_size[1])

            anchor_x = random.uniform(x_start, x_end)
            anchor_y = random.uniform(y_start, y_end)

            anchor_x, anchor_y = int(anchor_x), int(anchor_y)

            low_dose_ct = low_dose_ct[anchor_x:anchor_x + dim_x, anchor_y:anchor_y + dim_y]
            normal_dose_ct = normal_dose_ct[anchor_x:anchor_x + dim_x, anchor_y:anchor_y + dim_y]

        low_dose_ct = self._normalization(low_dose_ct)
        low_dose_ct = torch.from_numpy(low_dose_ct).unsqueeze(0)

        normal_dose_ct = self._normalization(normal_dose_ct)
        normal_dose_ct = torch.from_numpy(normal_dose_ct).unsqueeze(0)

        return {'input': low_dose_ct,
                'target': normal_dose_ct,
                'current_data_path': low_dose_ct_path}

    def __len__(self):
        return self.dataset_size


def get_data_provider(cfg, phase='Train'):
    is_gpu = torch.cuda.is_available()
    is_train = True if phase.upper() == 'TRAIN' else False
    is_shuffle = True if phase.upper() == 'TRAIN' else False

    target_size = cfg.MODEL.GENERATOR.INPUT_SIZE
    shift_size = cfg.MODEL.GENERATOR.SHIFT_SIZE

    data_list = {'TRAIN': cfg.DATASET.TRAIN_LIST,
                 'VAL': cfg.DATASET.VAL_LIST,
                 'TEST': cfg.DATASET.TEST_LIST}

    batch_size = eval('cfg.{}.BATCHSIZE_PER_GPU * torch.cuda.device_count()'.format(phase.upper()))
    iteration = int(cfg.TRAIN.TOTAL_ITERATION)

    current_dataset = AAPMLowDose(crop_size=target_size,
                                  shift_size=shift_size,
                                  data_root=cfg.DATASET.ROOT,
                                  data_list=data_list[phase.upper()],
                                  is_train=is_train,
                                  customized_dataset_size=batch_size * iteration if phase.upper() == 'TRAIN' else 0
                                  )

    data_loader = torch.utils.data.DataLoader(current_dataset,
                                              batch_size=batch_size,
                                              shuffle=is_shuffle,
                                              num_workers=cfg.WORKERS,
                                              pin_memory=is_gpu)

    return data_loader


if __name__ == '__main__':
    import matplotlib

    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    target_size = (512, 512)
    shift_size = (16, 16)
    data_root = r'D:\share\data\Challenge\LowDoseChanllenge\numpy'
    data_list = 'train.list'
    is_train = True
    is_shuffle = True

    batch_size = 2
    num_threads = 0
    is_gpu = torch.cuda.is_available()

    current_dataset = AAPMLowDose(crop_size=target_size,
                                  shift_size=shift_size,
                                  data_root=data_root,
                                  data_list=data_list,
                                  is_train=is_train)

    train_loader = torch.utils.data.DataLoader(current_dataset,
                                               batch_size=batch_size,
                                               shuffle=is_shuffle,
                                               num_workers=num_threads,
                                               pin_memory=is_gpu)

    for i, data in enumerate(train_loader):
        noisy_input, clean_target = data['input'], data['target']
        noisy_input = noisy_input.squeeze().detach().cpu().numpy()[0]
        clean_target = clean_target.squeeze().detach().cpu().numpy()[0]

        print("iter {}, "
              "input min/max: {}/{}, "
              "target min/max: {}/{}".format(i,
                                             np.min(noisy_input), np.max(noisy_input),
                                             np.min(clean_target), np.max(clean_target)))

        plt.figure()
        plt.subplot(1, 3, 1)
        plt.imshow(noisy_input, cmap='gray', vmin=0, vmax=1.0)
        plt.subplot(1, 3, 2)
        plt.imshow(clean_target, cmap='gray', vmin=0, vmax=1.0)
        plt.subplot(1, 3, 3)
        plt.imshow(noisy_input - clean_target, cmap='gray', vmin=-0.05, vmax=0.05)
        plt.show()

    print('Congrats! May the force be with you ...')
