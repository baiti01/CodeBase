#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MIAI Lab | UT Southwestern Medical Center
# DATETIME: 1/7/2020 3:23 PM

# sys
import os
from skimage.measure import compare_ssim
import numpy as np

# torch
import torch

# project
from lib.analyze.base_visualize import VisualizationAnalysis


class VolumeVisualization(object):
    def __init__(self,
                 data_range=[-1, 1],
                 before_min=0,
                 before_max=2000,
                 after_min=-1,
                 after_max=1,
                 threshold=200
                 ):
        self.data_path = []
        self.input = []
        self.output = []
        self.target = []
        self.rmse = []
        self.ssim = []
        self.patient = {}
        self.data_range = data_range[1] - data_range[0]
        self.threshold = (threshold - before_min) / (before_max - before_min) * (after_max - after_min) + after_min

    def update(self, current_data, current_model):
        rmse = []
        ssim = []
        for i in range(len(current_data['current_data_path'])):
            current_input = current_model.input.detach()[i].unsqueeze(0)
            current_target = current_model.target.detach()[i].unsqueeze(0)
            current_output = current_model.output.detach()[i].unsqueeze(0)
            current_data_path = current_data['current_data_path'][i]

            self.data_path.append(current_data_path)
            self.input.append(current_input)
            self.target.append(current_target)
            self.output.append(current_output)

            current_rmse, current_ssim = self.similarity_2D_calculation(current_output, current_target)
            rmse.append(current_rmse)
            ssim.append(current_ssim)
        self.rmse += rmse
        self.ssim += ssim
        return np.mean(np.array(rmse)), np.mean(np.array(ssim))

    def volumize(self):
        for idx, current_path in enumerate(self.data_path):
            current_name = os.path.basename(current_path)[:4]
            if current_name not in self.patient:
                self.patient[current_name] = {}
                self.patient[current_name]['slice_index'] = [idx]
            else:
                self.patient[current_name]['slice_index'].append(idx)

        key = 0
        for k, v in self.patient.items():
            key = k
            start_slice = v['slice_index'][0]
            end_slice = v['slice_index'][-1] + 1
            v['input'] = torch.stack(self.input[start_slice:end_slice], dim=-1)
            v['output'] = torch.stack(self.output[start_slice:end_slice], dim=-1)
            v['target'] = torch.stack(self.target[start_slice:end_slice], dim=-1)

        self.input = self.patient[key]['input']
        self.output = self.patient[key]['output']
        self.target = self.patient[key]['target']

    def similarity_2D_calculation(self, current_output, current_target):
        current_output = current_output.squeeze()
        current_target = current_target.squeeze()

        effective_region = torch.ge(current_target, self.threshold).float()

        # rmse calculation
        current_mse = torch.nn.functional.mse_loss(current_output, current_target, reduce=False)
        current_mse = effective_region * current_mse
        current_rmse = torch.sqrt(torch.sum(current_mse) / torch.sum(effective_region)).item()
        current_ssim, current_ssim_map = compare_ssim(current_output.cpu().numpy(),
                                                      current_target.cpu().numpy(),
                                                      data_range=self.data_range,
                                                      full=True)
        current_effective_ssim = current_ssim_map * effective_region.cpu().numpy()
        current_effective_ssim = np.sum(current_effective_ssim) / torch.sum(effective_region).item()
        return current_rmse, current_effective_ssim

    def similarity_3D_calculation(self):
        result_rmse = []
        result_ssim = []
        for k, v in self.patient.items():
            body_contour = torch.ge(v['target'][0][0], self.threshold).float()

            # calculate the mse
            current_mse = torch.nn.functional.mse_loss(v['output'][0][0], v['target'][0][0], reduce=False)
            current_mse = body_contour * current_mse
            current_rmse = torch.sqrt(torch.sum(current_mse) / torch.sum(body_contour)).item()

            current_ssim, current_ssim_map = compare_ssim(v['output'][0][0].cpu().numpy(),
                                                          v['target'][0][0].cpu().numpy(),
                                                          data_range=self.data_range,
                                                          full=True)
            current_effective_ssim = current_ssim_map * body_contour.cpu().numpy()
            current_effective_ssim = np.sum(current_effective_ssim) / torch.sum(body_contour).item()
            
            result_ssim.append(current_effective_ssim)
            result_rmse.append(current_rmse)
        return result_rmse, result_ssim


def visualize(volume_visualizer,
              model,
              iteration,
              output_dir,
              display_frequency,
              before_min=0,
              before_max=2000,
              after_min=-1,
              after_max=1,
              window_size=256):
    # update website
    volume_visualizer.volumize()
    rmse, ssim = volume_visualizer.similarity_3D_calculation()
    visualizer = VisualizationAnalysis(model, iteration, output_dir)
    webpage = visualizer.webpage
    scale = (before_max - before_min) / (after_max - after_min)
    for current_image_name in model.visualized_images:
        if current_image_name == 'diff':
            shift = 0
            current_image_tensor = volume_visualizer.target - volume_visualizer.output
            min_v, max_v = -50, 50
        else:
            shift = before_min - after_min * scale
            current_image_tensor = getattr(volume_visualizer, current_image_name)
            min_v, max_v = 1000 - 160, 1000 + 240

        image_numpy = visualizer.tensor2image(current_image_tensor, scale, shift)
        image_numpy.tofile(os.path.join(output_dir, 'web', 'images', '{}.bin'.format(current_image_name)))

        img_path = 'iter_{:>08}_{}'.format(iteration, current_image_name)
        visualizer.save_image(image_numpy, os.path.join(output_dir, 'web', 'images', img_path), min_v, max_v)

    for n in range(iteration, -1, -display_frequency):
        webpage.add_header('iteration_{:>08}'.format(n))
        ims, txts, links = [], [], []
        for current_image_name in model.visualized_images:
            if 'supervised' in current_image_name:
                pass
            for postfix in ['_t.png', '_c.png', '_s.png']:
                img_path = 'iter_{:>08}_{}{}'.format(n, current_image_name, postfix)
                ims.append(img_path)
                txts.append(current_image_name)
                links.append(img_path)
        webpage.add_images(ims, txts, links, window_size)
    webpage.save()

    return rmse, ssim


if __name__ == '__main__':
    print('Congrats! May the force be with you ...')
