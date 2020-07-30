#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MIAI Lab | UT Southwestern Medical Center
# DATETIME: 12/18/2019 2:27 PM

# torch
import torch

# project
from lib.model.utils.utils import network_initialization


def define_generator(generator_config):
    generator_name = generator_config.NAME
    if 'unet' in generator_name.lower():
        from lib.model.generator.unet_generator import UnetGenerator
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
        from lib.model.generator.plain_generator import PlainGenerator
        network = PlainGenerator(input_channels,
                                 output_channels,
                                 is_weight_standardization=is_weight_standardization,
                                 num_filters=filter_number_last_conv_layer,
                                 norm_layer=norm_layer,
                                 output_activation_layer=output_activation_layer)
    elif 'hrnet' in generator_name.lower():
        from lib.model.generator.hr_generator_basic import HighResolutionNet
        network = HighResolutionNet(cfg,
                                    is_weight_standardization=is_weight_standardization,
                                    output_activation_layer=output_activation_layer)
        network.init_weights()
    elif 'm3net' in generator_name.lower():
        from lib.model.generator.M3Net import DoseNet
        network = DoseNet(input_channels=generator_config.INPUT_CHANNELS,
                          feature_channels=generator_config.FEATURE_CHANNELS,
                          layer_numbers=generator_config.LAYER_NUMBERS,
                          prediction_head_channels=generator_config.PREDICTION_HEAD_CHANNELS,
                          scale_dimension=generator_config.SCALE_DIMENSION,
                          input_dimension=generator_config.INPUT_SIZE)
        network = network_initialization(network,
                                         init_type=generator_config.INIT_TYPE,
                                         init_gain=generator_config.INIT_GAIN)
    else:
        raise('Unsupported network: {}'.format(generator_name))

    return network


def define_discriminator(input_channels,
                         filter_number_first_conv_layer,
                         network_name,
                         number_conv_layers=3,
                         norm='batch',
                         dimension='2d',
                         init_type='normal',
                         init_gain=0.02):
    norm_layer = get_norm_layer(norm_type=norm)

    if network_name == 'basic':  # default PatchGAN classifier
        from lib.model.discriminator.NLayer_discriminator import NLayerDiscriminator
        net = NLayerDiscriminator(input_channels,
                                  dimension,
                                  filter_number_first_conv_layer,
                                  number_conv_layers=3,
                                  norm_layer=norm_layer)
    elif network_name == 'n_layers':  # more options
        from lib.model.discriminator.NLayer_discriminator import NLayerDiscriminator
        net = NLayerDiscriminator(input_channels,
                                  dimension,
                                  filter_number_first_conv_layer,
                                  number_conv_layers,
                                  norm_layer=norm_layer)
    elif network_name == 'pixel':  # classify if each pixel is real or fake
        from lib.model.discriminator.pixel_discriminator import PixelDiscriminator
        net = PixelDiscriminator(input_channels,
                                 dimension,
                                 filter_number_first_conv_layer,
                                 norm_layer=norm_layer)
    else:
        raise NotImplementedError('Discriminator loss name {} is not recognized'.format(network_name))
    return network_initialization(net, init_type, init_gain)


if __name__ == '__main__':
    print('Congrats! May the force be with you ...')
