#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 9/27/2019 12:00 PM
# FILE: visualize.py

# sys
import os

# project
from lib.analyze.base_visualize import VisualizationAnalysis


def visualize(model,
              iteration,
              output_dir,
              display_frequency,
              window_size=256):
    # update website
    visualizer = VisualizationAnalysis(model, iteration, output_dir)
    webpage = visualizer.webpage

    before_min = 0
    before_max = 2000
    after_min = 0
    after_max = 1
    scale = (before_max - before_min) / (after_max - after_min)

    for current_image_name in model.visualized_images:
        if current_image_name == 'diff':
            shift = 0
            current_image_tensor = model.target - model.output
            min_v, max_v = -50, 50
        else:
            shift = before_min - after_min * scale
            current_image_tensor = getattr(model, current_image_name)
            min_v, max_v = 1000-160, 1000+240

        image_numpy = visualizer.tensor2image(current_image_tensor, scale, shift)
        img_path = 'iter_{:>08}_{}'.format(iteration, current_image_name)
        visualizer.save_image(image_numpy, os.path.join(output_dir, 'web', 'images', img_path), min_v, max_v, cmap='gray')

    for n in range(iteration, -1, -display_frequency):
        webpage.add_header('iteration_{:>08}'.format(n))
        ims, txts, links = [], [], []
        for current_image_name in model.visualized_images:
            for postfix in ['_c.png', '_t.png', '_s.png']:
            #for postfix in ['.png']:
                img_path = 'iter_{:>08}_{}{}'.format(n, current_image_name, postfix)
                ims.append(img_path)
                txts.append(current_image_name)
                links.append(img_path)
        webpage.add_images(ims, txts, links, window_size)
    webpage.save()
