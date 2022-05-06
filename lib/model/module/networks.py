#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 12/18/2019 2:27 PM

# torch
import torch


def define_generator(generator_config):
    generator_name = generator_config.NAME
    if 'unet_monai' == generator_name.lower():
        from monai.networks.nets import UNet
        network = UNet(dimensions=generator_config.DIMENSIONS,
                       in_channels=generator_config.INPUT_CHANNELS,
                       out_channels=generator_config.OUTPUT_CHANNELS,
                       channels=generator_config.CHANNELS,
                       strides=generator_config.STRIDES,
                       num_res_units=generator_config.NUM_RES_UNITS)
    elif 'lightweight_UNet'.lower() == generator_name.lower():
        from lib.model.module.Lightweight_Unet import UnetGenerator
        network = UnetGenerator(input_channels=generator_config.INPUT_CHANNELS,
                                output_channels=generator_config.OUTPUT_CHANNELS,
                                downsampling_number=generator_config.DOWNSAMPLING_NUMBER,
                                filter_number_last_conv_layer=generator_config.FILTER_NUMBER_LAST_CONV_LAYER,
                                norm_layer=generator_config.NORM_LAYER,
                                dimension=generator_config.DIMENSION)
    else:
        raise ('Unsupported network: {}'.format(generator_name))
    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    return network


def def_discriminator(discriminator_config):
    raise NotImplementedError('Not implemented yet!')


if __name__ == '__main__':
    print('Congrats! May the force be with you ...')
