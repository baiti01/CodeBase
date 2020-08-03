#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 7/14/2020 5:18 PM

# sys
import numpy as np
import os
import cv2
from PIL import Image
import random
import logging

# torch
import torch

# project
from lib.dataset.base_dataset import BaseDataset


logger = logging.getLogger(__name__)
CheXpert_disease = ['No Finding',
                    'Enlarged Cardiomediastinum',
                    'Cardiomegaly',
                    'Lung Opacity',
                    'Lung Lesion',
                    'Edema',
                    'Consolidation',
                    'Pneumonia',
                    'Atelectasis',
                    'Pneumothorax',
                    'Pleural Effusion',
                    'Pleural Other',
                    'Fracture',
                    'Support Devices']


class MIMIC(BaseDataset):
    def __init__(self, data_root, label_path, hyper_parameters, customized_dataset_size=0, mode='train'):
        super(MIMIC, self).__init__()
        self.pixel_mean = hyper_parameters['pixel_mean']
        self.pixel_std = hyper_parameters['pixel_std']
        self.long_side = hyper_parameters['long_side']
        self.border_type = hyper_parameters['border_type']
        self.gaussian_blur = hyper_parameters['gaussian_blur']

        self.number_class = hyper_parameters['number_class']
        self.ignore_index = hyper_parameters['ignore_index']
        self.unsure_label = hyper_parameters['unsure_label']

        assert (self.number_class == 2 and self.unsure_label != 2) or (
                    self.number_class == 3 and self.unsure_label == 2)

        self._mode = mode
        self._label_header = None
        self.path_labels = []

        self._image_paths = []
        self._labels = []

        self.dict = [{'1': '1', '': str(self.ignore_index), '0': '0', '-1': str(self.unsure_label)},
                     {'1': '1', '': str(self.ignore_index), '0': '0', '-1': str(self.ignore_index)}]
        with open(os.path.join(data_root, label_path)) as f:
            self._label_header = f.readline().strip('\n').split(',')
            self.disease_name = self._label_header[6:]

            # reorder the disease as the CheXpert
            self.disease_index = []
            for current_disease_name in CheXpert_disease:
                self.disease_index.append(self.disease_name.index(current_disease_name))
            for line in f:
                fields = line.strip('\n').split(',')
                subject_id, study_id, dicom_id, sex, split, view = fields[:6]
                image_path = os.path.join(data_root,
                                          'p{}'.format(subject_id[:2]),
                                          'p{}'.format(subject_id),
                                          's{}'.format(study_id),
                                          '{}.jpg'.format(dicom_id))
                current_labels = fields[6:]
                current_labels = [current_labels[i] for i in self.disease_index]

                no_finding = current_labels[0]
                if no_finding == '1.0' or no_finding == '1':
                    no_finding = '1'
                else:
                    no_finding = '0'

                if self._mode == 'train':
                    labels = [self.dict[0].get(n) for n in current_labels[1:]]
                else:
                    labels = [self.dict[1].get(n) for n in current_labels[1:]]
                labels = [no_finding] + labels
                labels = list(map(int, labels))
                self.path_labels.append((image_path, labels))

        self._num_image = len(self.path_labels)
        self.dataset_size = customized_dataset_size if customized_dataset_size else self._num_image

    def __len__(self):
        return self.dataset_size

    def __getitem__(self, idx):
        idx = idx % self._num_image
        current_image_path, current_label = self.path_labels[idx]
        current_image = cv2.imread(current_image_path, 0)

        # if cannot read this image correctly, choose another one
        if current_image is None:
            logger.info('Image None: {}!'.format(current_image_path))
            for i in range(100):
                idx = random.randint(0, self._num_image - 1)
                current_image = cv2.imread(self.path_labels[idx][0], 0)
                if current_image is not None:
                    break

        if self._mode == 'train':
            current_image = Image.fromarray(current_image)
            current_image = self.Aug(current_image)
            current_image = np.array(current_image)

        current_image = self.Common(current_image)
        current_image = self._fix_ratio(current_image, border_type=self.border_type, long_side=self.long_side)

        # normalization
        current_image = (current_image - self.pixel_mean) / self.pixel_std

        current_image = current_image.transpose((2, 0, 1))
        current_label = np.array(current_label).astype(np.int64)

        return {'input': current_image,
                'target': current_label,
                'path': current_image_path}


def get_data_provider(cfg, phase='Train'):
    is_gpu = torch.cuda.is_available()
    is_shuffle = True if phase.upper() == 'TRAIN' else False

    batch_size = eval('cfg.{}.BATCHSIZE_PER_GPU * torch.cuda.device_count()'.format(phase.upper()))

    iteration = int(cfg.TRAIN.TOTAL_ITERATION)
    hyper_parameters = {'pixel_mean': cfg.DATASET.PIXEL_MEAN,
                        'pixel_std': cfg.DATASET.PIXEL_STD,
                        'long_side': cfg.DATASET.LONG_SIDE,
                        'border_type': cfg.DATASET.BORDER_TYPE,
                        'gaussian_blur': cfg.DATASET.GAUSSIAN_BLUR,
                        'number_class': cfg.MODEL.NUM_CLASSES,
                        'ignore_index': cfg.MODEL.IGNORE_CLASS,
                        'unsure_label': cfg.MODEL.UNSURE_LABEL}

    data_list = {'train': cfg.DATASET.TRAIN_LIST,
                 'val': cfg.DATASET.VAL_LIST,
                 'test': cfg.DATASET.TEST_LIST}

    current_dataset = MIMIC(data_root=cfg.DATASET.ROOT,
                            label_path=data_list[phase.lower()],
                            hyper_parameters=hyper_parameters,
                            customized_dataset_size=batch_size * iteration if phase.upper() == 'TRAIN' else 0,
                            mode=phase.lower())

    data_loader = torch.utils.data.DataLoader(current_dataset,
                                              batch_size=batch_size,
                                              shuffle=is_shuffle,
                                              num_workers=cfg.WORKERS,
                                              pin_memory=is_gpu)

    return data_loader
