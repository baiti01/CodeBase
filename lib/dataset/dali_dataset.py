#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 10/10/2019 11:39 AM
# FILE: main.py

# sys
import os
import numpy as np
import random
from random import shuffle

# torch
import torch

# nvidia
from nvidia.dali.pipeline import Pipeline
import nvidia.dali.ops as ops
from nvidia.dali.plugin.pytorch import DALIGenericIterator as PyTorchIterator


class ExternalInputIterator(object):
    def __init__(self, batch_size,
                 crop_size=(224, 224, 48),
                 shift_size=(16, 16, 8),
                 data_root=r'./data',
                 is_train=True):

        if shift_size:
            if isinstance(shift_size, int):
                shift_size = (shift_size, shift_size, shift_size)
            assert isinstance(shift_size, tuple)
        else:
            shift_size = 'maximum'

        self.shift_size = shift_size
        self.batch_size = batch_size
        self.crop_size = crop_size
        self.is_train = is_train

        self.dose_type = ['1e6', '1e7', '1e8']
        self.patient_mrn = []
        for roots, dirs, files in os.walk(data_root):
            for current_file in files:
                print(current_file)
                current_mrn = current_file.split('_')[0]
                self.patient_mrn.append(current_mrn)

        self.patient_mrn = list(set(self.patient_mrn))

        self.dose = {}
        for idx, current_patient in enumerate(self.patient_mrn):
            self.dose[current_patient] = {}
            for current_dose_type in self.dose_type:
                current_dose_file = current_patient + '_' + current_dose_type + '.bin'
                current_dose_path = os.path.join(data_root, current_dose_file)
                tmp = np.fromfile(current_dose_path, dtype=np.float32).reshape(64, 256, 256).transpose((1, 2, 0))
                self.dose[current_patient][current_dose_type] = np.ascontiguousarray(tmp)

        self.n = len(self.patient_mrn)

    def __iter__(self):
        self.i = 0
        if self.is_train:
            shuffle(self.patient_mrn)
        return self

    @staticmethod
    def __calculate_anchor_location__(origin_size, target_size, target_shift_size):
        if target_shift_size * 2 > origin_size - target_size:
            target_shift_size = (origin_size - target_size) / 2

        start = ((origin_size / 2 - target_shift_size) - target_size / 2) / origin_size
        end = ((origin_size / 2 + target_shift_size) - target_size / 2) / origin_size
        return start, end

    def __next__(self):
        noisy_list = []
        clean_list = []
        anchor_list = []
        slice_list = []
        if self.i >= self.n:
            raise StopIteration

        for _ in range(self.batch_size):
            noisy_input = self.dose[self.patient_mrn[self.i]]['1e6']
            clean_target = self.dose[self.patient_mrn[self.i]]['1e8']

            # calculate the slice size
            noisy_input_x, noisy_input_y, noisy_input_z = noisy_input.shape
            dim_x, dim_y, dim_z = self.crop_size
            slice_x, slice_y, slice_z = dim_x / noisy_input_x, dim_y / noisy_input_y, dim_z / noisy_input_z

            # calculate the anchor location
            if self.shift_size is 'maximum':
                x_start, x_end = 0.0, 1 - slice_x
                y_start, y_end = 0.0, 1 - slice_y
                z_start, z_end = 0.0, 1 - slice_z
            else:
                x_start, x_end = self.__calculate_anchor_location__(noisy_input_x, dim_x, self.shift_size[0])
                y_start, y_end = self.__calculate_anchor_location__(noisy_input_y, dim_y, self.shift_size[1])
                z_start, z_end = self.__calculate_anchor_location__(noisy_input_z, dim_z, self.shift_size[2])

            if self.is_train:
                anchor_x = random.uniform(x_start, x_end)
                anchor_y = random.uniform(y_start, y_end)
                anchor_z = random.uniform(z_start, z_end)
            else:
                anchor_x = (x_start + x_end) / 2.0
                anchor_y = (y_start + y_end) / 2.0
                anchor_z = (z_start + z_end) / 2.0

            slice_list.append(np.array([slice_x, slice_y, slice_z], dtype=np.float32))
            anchor_list.append(np.array([anchor_x, anchor_y, anchor_z], dtype=np.float32))
            noisy_list.append(noisy_input)
            clean_list.append(clean_target)

            self.i = (self.i + 1) % self.n
        return noisy_list, clean_list, anchor_list, slice_list

    @property
    def size(self, ):
        return len(self.patient_mrn)

    next = __next__


class ExternalSourcePipeline(Pipeline):
    def __init__(self, batch_size, num_threads, device_id, external_data, is_train=True):
        super(ExternalSourcePipeline, self).__init__(batch_size, num_threads, device_id, seed=12)
        self.is_train = is_train
        self.input = ops.ExternalSource()

        self.angle_rng = ops.Uniform(range=(-10.0, 10.0))
        self.rotate = ops.Rotate(device="gpu")

        self.flip_rng = ops.CoinFlip(probability=0.5)
        self.flip = ops.Flip(device='gpu')

        self.slice = ops.Slice()
        self.external_data = external_data
        self.iterator = iter(self.external_data)

    def define_graph(self):
        # define the input
        self.noisy_input = self.input()
        self.clean_target = self.input()
        self.anchor_location = self.input()
        self.slice_size = self.input()
        noisy_input, clean_target, anchor_location, slice_size = self.noisy_input, self.clean_target, self.anchor_location, self.slice_size

        transformed_noisy_input = self.slice(noisy_input, anchor_location, slice_size)
        transformed_noisy_input = transformed_noisy_input.gpu()

        transformed_clean_target = self.slice(clean_target, anchor_location, slice_size)
        transformed_clean_target = transformed_clean_target.gpu()

        # data augmentation
        if self.is_train:
            # random rotation and flip
            angle = self.angle_rng()
            flip = self.flip_rng()

            transformed_noisy_input = self.rotate(transformed_noisy_input, angle=angle)
            transformed_noisy_input = self.flip(transformed_noisy_input, horizontal=flip)

            transformed_clean_target = self.rotate(transformed_clean_target, angle=angle)
            transformed_clean_target = self.flip(transformed_clean_target, horizontal=flip)

        return transformed_noisy_input, transformed_clean_target

    def iter_setup(self):
        try:
            (noisy_input, clean_target, anchor_position, slice_size) = self.iterator.next()
            self.feed_input(self.noisy_input, noisy_input)
            self.feed_input(self.clean_target, clean_target)
            self.feed_input(self.anchor_location, anchor_position)
            self.feed_input(self.slice_size, slice_size)
        except StopIteration:
            self.iterator = iter(self.external_data)
            raise StopIteration


def get_data_provider(cfg, is_train=True, device_id=0):
    target_size = cfg.MODEL.GENERATOR.A.INPUT_SIZE
    shift_size = cfg.MODEL.GENERATOR.A.SHIFT_SIZE
    if is_train:
        data_root = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.TRAIN_SET)
        batch_size = cfg.TRAIN.BATCHSIZE_PER_GPU * torch.cuda.device_count()
    else:
        data_root = os.path.join(cfg.DATASET.ROOT, cfg.DATASET.VAL_SET)
        batch_size = cfg.VAL.BATCHSIZE_PER_GPU * torch.cuda.device_count()

    num_threads = cfg.WORKERS

    eii = ExternalInputIterator(batch_size=batch_size,
                                crop_size=target_size,
                                shift_size=shift_size,
                                data_root=data_root,
                                is_train=is_train)
    pipe = ExternalSourcePipeline(batch_size=batch_size,
                                  num_threads=num_threads,
                                  device_id=device_id,
                                  external_data=eii,
                                  is_train=is_train)
    pii = PyTorchIterator(pipe,
                          size=eii.size,
                          auto_reset=True,
                          dynamic_shape=True,
                          output_map=['noisy_input', 'clean_target'])
    return pii


if __name__ == '__main__':
    import matplotlib.pyplot as plt

    xx = os.getcwd()
    target_size = (224, 224, 48)
    shift_size = (16, 16, 8)
    data_root = './data'
    is_train = True
    batch_size = 1
    epochs = 300

    pii = get_data_provider(target_size=target_size,
                            shift_size=shift_size,
                            data_root=data_root,
                            is_train=is_train,
                            batch_size=batch_size)

    for e in range(epochs):
        for i, data in enumerate(pii):
            print("epoch: {}, iter {}, real batch size: {}".format(e, i, data[0]["noisy_input"].shape))
            ct, mr = data[0]['noisy_input'], data[0]['clean_target']

            if True:
                plt.figure()
                plt.subplot(2, 3, 1)
                plt.imshow(data[0]['noisy_input'][0, :, :, 24].cpu().numpy(), cmap='jet', vmin=0, vmax=80)
                plt.subplot(2, 3, 2)
                plt.imshow(data[0]['noisy_input'][0, 112, :, :].cpu().numpy(), cmap='jet', vmin=0, vmax=80)
                plt.subplot(2, 3, 3)
                plt.imshow(data[0]['noisy_input'][0, :, 112, :].cpu().numpy(), cmap='jet', vmin=0, vmax=80)
                plt.subplot(2, 3, 4)
                plt.imshow(data[0]['clean_target'][0, :, :, 24].cpu().numpy(), cmap='jet', vmin=0, vmax=80)
                plt.subplot(2, 3, 5)
                plt.imshow(data[0]['clean_target'][0, :, 112, :].cpu().numpy(), cmap='jet', vmin=0, vmax=80)
                plt.subplot(2, 3, 6)
                plt.imshow(data[0]['clean_target'][0, 112, :, :].cpu().numpy(), cmap='jet', vmin=0, vmax=80)
                plt.savefig(r'vis/vis_{}_{}.png'.format(e, i))
        pii.reset()

    print('Congrats! May the force be with you ...')
