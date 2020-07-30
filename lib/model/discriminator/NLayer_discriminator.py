#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MIAI Lab | UT Southwestern Medical Center
# DATETIME: 12/18/2019 3:18 PM

# sys
import functools

# torch
import torch.nn as nn

# project
from lib.model.modules import net_dimension


class NLayerDiscriminator(nn.Module):
    """Defines a PatchGAN discriminator"""

    def __init__(self, input_channels,
                 dimension='2d',
                 filter_number_first_conv_layer=64,
                 number_conv_layers=3,
                 norm_layer=nn.BatchNorm2d):
        """Construct a PatchGAN discriminator

        Parameters:
            input_channels (int)  -- the number of channels in input images
            dimension (str) -- 2d | 3d
            filter_number_first_conv_layer (int)       -- the number of filters in the last conv layer
            number_conv_layers (int)  -- the number of conv layers in the discriminator
            norm_layer      -- normalization layer
        """
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = (norm_layer.func != nn.BatchNorm2d) or ((norm_layer.func != nn.BatchNorm3d))
        else:
            use_bias = (norm_layer != nn.BatchNorm2d) or (norm_layer != nn.BatchNorm3d)

        conv_layer, _ = net_dimension(dimension)

        kw = 4
        padw = 1
        sequence = [conv_layer(input_channels,
                               filter_number_first_conv_layer,
                               kernel_size=kw,
                               stride=2,
                               padding=padw),
                    nn.LeakyReLU(0.2, True)]

        nf_mult = 1
        nf_mult_prev = 1
        for n in range(1, number_conv_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                conv_layer(filter_number_first_conv_layer * nf_mult_prev,
                           filter_number_first_conv_layer * nf_mult,
                           kernel_size=kw,
                           stride=2,
                           padding=padw,
                           bias=use_bias),
                norm_layer(filter_number_first_conv_layer * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** number_conv_layers, 8)
        sequence += [
            conv_layer(filter_number_first_conv_layer * nf_mult_prev,
                       filter_number_first_conv_layer * nf_mult,
                       kernel_size=kw,
                       stride=1,
                       padding=padw,
                       bias=use_bias),
            norm_layer(filter_number_first_conv_layer * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        # output 1 channel prediction map
        sequence += [
            conv_layer(filter_number_first_conv_layer * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)


if __name__ == '__main__':
    print('Congrats! May the force be with you ...')
