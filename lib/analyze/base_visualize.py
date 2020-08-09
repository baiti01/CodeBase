#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MIAI Lab | UT Southwestern Medical Center
# DATETIME: 12/30/2019 6:04 PM

# sys
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

# project
from lib.analyze import html


class VisualizationAnalysis(object):
    def __init__(self, model, iteration, output_dir):
        self.model = model
        self.iteration = iteration
        self.output_dir = output_dir
        self.webpage = html.HTML(os.path.join(output_dir, 'web'),
                                 'Experiment name = {}'.format(os.path.basename(output_dir)),
                                 refresh=60)

    @staticmethod
    def save_single_image(image_numpy, saved_image_path, min_v=0, max_v=2, cmap='gray'):
        plt.figure()
        plt.imshow(image_numpy, cmap=cmap, vmin=min_v, vmax=max_v)
        plt.axis('off')
        plt.colorbar()
        plt.savefig(saved_image_path, bbox_inches='tight')
        plt.close()

    def save_image(self, image_numpy, image_path, min_v=0, max_v=2, cmap='gray'):
        if len(image_numpy.shape) == 2:
            saved_image_path = image_path + '.png'
            self.save_single_image(image_numpy, saved_image_path, min_v, max_v, cmap)
        else:
            assert len(image_numpy.shape) == 3
            h, w, d = image_numpy.shape
            saved_image_path = image_path + '_t.png'
            self.save_single_image(image_numpy[:, :, int(d / 2)], saved_image_path, min_v, max_v, cmap)

            saved_image_path = image_path + '_s.png'
            self.save_single_image(image_numpy[:, int(w / 2), :].transpose((1, 0)), saved_image_path, min_v, max_v, cmap)

            saved_image_path = image_path + '_c.png'
            self.save_single_image(image_numpy[int(h / 2), :, :].transpose((1, 0)), saved_image_path, min_v, max_v, cmap)

    def tensor2image(self, input_image, scale=1, shift=0, batch_index=0, channel_index=0):
        if not isinstance(input_image, np.ndarray):
            image_numpy = input_image[batch_index][channel_index].detach().cpu().numpy()
            # customized post-processing for scaled value, here, suppose the original output range is [-1, 1]
            image_numpy = image_numpy * scale + shift
        else:  # if it is a numpy array, do nothing
            image_numpy = input_image

        return image_numpy

    def visualize(self):
        pass
