#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 10/14/2019 4:16 PM
# FILE: modules.py

# sys
import random

# torch
import torch
import torch.nn as nn


def pixel_unshuffle(x, upscale_factor=2):
    dimension = 2 if len(x.shape) == 4 else 3
    if dimension == 2:
        b, c, h, w = x.shape
        c_out, h_out, w_out = c * (upscale_factor ** dimension), h // upscale_factor, w // upscale_factor
        x = x.contiguous().view(b, c, h_out, upscale_factor, w_out, upscale_factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, c_out, h_out, w_out)
    else:
        b, c, d, h, w = x.shape
        c_out, d_out, h_out, w_out = c * (upscale_factor ** dimension), d // upscale_factor, h // upscale_factor, w // upscale_factor
        x = x.contiguous().view(b, c, d_out, upscale_factor, h_out, upscale_factor, w_out, upscale_factor)
        x = x.permute(0, 1, 3, 5, 7, 2, 4, 6).contiguous().view(b, c_out, d_out, h_out, w_out)
    return x


def pixel_shuffle(input, upscale_factor=2):
    input_size = list(input.size())
    dimensionality = len(input_size) - 2
    input_size[1] //= (upscale_factor ** dimensionality)
    output_size = [dim * upscale_factor for dim in input_size[2:]]

    input_view = input.contiguous().view(input_size[0],
                                         input_size[1],
                                         *([upscale_factor] * dimensionality),
                                         *(input_size[2:]))

    indicies = [5, 2, 6, 3, 7, 4][::-1] if dimensionality == 3 else [4, 2, 5, 3][::-1]
    shuffle_out = input_view.permute(0, 1, *(indicies[::-1])).contiguous()

    return shuffle_out.view(input_size[0], input_size[1], *output_size)


def conv_bn_relu(in_channels, out_channels, kernel_size=3, stride=1, dilation=1, groups=1,
                 norm_type='batch', is_bias=False, is_3d=True, is_depthwise=False):
    conv_module = []
    padding = int((kernel_size - 1) / 2)
    if is_3d:
        conv = nn.Conv3d
        if norm_type == 'batch':
            norm = nn.BatchNorm3d
        else:
            norm = nn.InstanceNorm3d
            is_bias = True
    else:
        conv = nn.Conv2d
        if norm_type == 'batch':
            norm = nn.BatchNorm2d
        else:
            norm = nn.InstanceNorm2d
            is_bias = True

    if is_depthwise:
        conv_module.append(
            conv(in_channels, in_channels, 3, stride, padding, dilation, groups=in_channels, bias=is_bias))
        conv_module.append(conv(in_channels, out_channels, 1, 1, 0, bias=is_bias))
    else:
        conv_module.append(
            conv(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias=is_bias))

    conv_module.append(norm(out_channels))
    conv_module.append(nn.ReLU(inplace=True))

    return nn.Sequential(*conv_module)


class ImagePool():
    """This class implements an image buffer that stores previously generated images.

    This buffer enables us to update discriminators using a history of generated images
    rather than the ones produced by the latest generators.
    """

    def __init__(self, pool_size):
        """Initialize the ImagePool class

        Parameters:
            pool_size (int) -- the size of image buffer, if pool_size=0, no buffer will be created
        """
        self.pool_size = pool_size
        if self.pool_size > 0:  # create an empty pool
            self.num_imgs = 0
            self.images = []

    def query(self, images):
        """Return an image from the pool.

        Parameters:
            images: the latest generated images from the generator

        Returns images from the buffer.

        By 50/100, the buffer will return input images.
        By 50/100, the buffer will return images previously stored in the buffer,
        and insert the current images to the buffer.
        """
        if self.pool_size == 0:  # if the buffer size is 0, do nothing
            return images
        return_images = []
        for image in images:
            image = torch.unsqueeze(image.data, 0)
            if self.num_imgs < self.pool_size:   # if the buffer is not full; keep inserting current images to the buffer
                self.num_imgs = self.num_imgs + 1
                self.images.append(image)
                return_images.append(image)
            else:
                p = random.uniform(0, 1)
                if p > 0.5:  # by 50% chance, the buffer will return a previously stored image, and insert the current image into the buffer
                    random_id = random.randint(0, self.pool_size - 1)  # randint is inclusive
                    tmp = self.images[random_id].clone()
                    self.images[random_id] = image
                    return_images.append(tmp)
                else:       # by another 50% chance, the buffer will return the current image
                    return_images.append(image)
        return_images = torch.cat(return_images, 0)   # collect all the images and return
        return return_images


if __name__ == '__main__':
    print('Congrats! May the force be with you ...')
