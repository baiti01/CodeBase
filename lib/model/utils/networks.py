#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 12/18/2019 2:27 PM

# torch
import torch

# project
from lib.model.utils.utils import network_initialization


def define_generator(generator_config):
    generator_name = generator_config.NAME
    if 'unet' in generator_name.lower():
        from lib.model.generator.unet import UnetGenerator
        network = UnetGenerator(input_channels=generator_config.INPUT_CHANNELS,
                                output_channels=generator_config.OUTPUT_CHANNELS,
                                downsampling_number=generator_config.DOWNSAMPLING_NUMBER,
                                filter_number_last_conv_layer=generator_config.FILTER_NUMBER_LAST_CONV_LAYER,
                                norm_layer=generator_config.NORMALIZATION_TYPE,
                                dimension=generator_config.DIMENSION)

        network = network_initialization(network,
                                         init_type=generator_config.INIT_TYPE,
                                         init_gain=generator_config.INIT_GAIN)
    elif 'plain' in generator_name.lower():
        from lib.model.generator.plainnet import PlainNet
        network = PlainNet(input_channels=generator_config.INPUT_CHANNELS,
                           output_channels=generator_config.OUTPUT_CHANNELS,
                           layer_number=generator_config.LAYER_NUMBER,
                           channel_number=generator_config.CHANNEL_NUMBER,
                           norm_type=generator_config.NORMALIZATION_TYPE)
    elif 'hrnet' in generator_name.lower():
        from lib.model.generator.hrnet import HighResolutionNet
        network = HighResolutionNet(generator_config)
        network.init_weights()
    else:
        raise('Unsupported network: {}'.format(generator_name))
    if torch.cuda.is_available():
        network = torch.nn.DataParallel(network)
    return network


def define_discriminator(discriminator_cfg):
    discriminator_name = discriminator_cfg.NAME
    from lib.config.discriminator_config import discriminator_dict
    if discriminator_name in discriminator_dict:
        network = discriminator_dict[discriminator_name](num_classes=discriminator_cfg.num_classes)
    elif discriminator_name == 'basic':  # default PatchGAN classifier
        from lib.model.discriminator.NLayer_discriminator import NLayerDiscriminator
        net = NLayerDiscriminator(input_channels,
                                  dimension,
                                  filter_number_first_conv_layer,
                                  number_conv_layers=3,
                                  norm_layer=norm_layer)
    elif discriminator_name == 'n_layers':  # more options
        from lib.model.discriminator.NLayer_discriminator import NLayerDiscriminator
        net = NLayerDiscriminator(input_channels,
                                  dimension,
                                  filter_number_first_conv_layer,
                                  number_conv_layers,
                                  norm_layer=norm_layer)
    elif discriminator_name == 'pixel':  # classify if each pixel is real or fake
        from lib.model.discriminator.pixel_discriminator import PixelDiscriminator
        net = PixelDiscriminator(input_channels,
                                 dimension,
                                 filter_number_first_conv_layer,
                                 norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator loss name {} is not recognized'.format(network_name))

    return network


if __name__ == '__main__':
    print('Congrats! May the force be with you ...')
