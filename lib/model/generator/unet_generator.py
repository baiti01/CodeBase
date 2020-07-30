#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MIAI Lab | UT Southwestern Medical Center
# DATETIME: 12/18/2019 3:10 PM

# sys
import functools

# torch
import torch
import torch.nn as nn


def net_dimension(dimension='2d'):
    if dimension == '2d':
        conv_layer = nn.Conv2d
        upsampling_mode = 'bilinear'
    elif dimension == '3d':
        conv_layer = nn.Conv3d
        upsampling_mode = 'trilinear'
    else:
        raise NotImplementedError('Unrecognized conv layer dimension: {}'.format(dimension))
    return conv_layer, upsampling_mode


def use_bias(norm_layer):
    if type(norm_layer) == functools.partial:
        is_bias = ((norm_layer.func != nn.BatchNorm2d) and (norm_layer.func != nn.BatchNorm3d))
    else:
        is_bias = ((norm_layer != nn.BatchNorm2d) and (norm_layer != nn.BatchNorm3d))
    return is_bias


class Identity(nn.Module):
    def forward(self, x):
        return x


def get_norm_layer(norm_type='instance', dimension='2d'):
    if norm_type == 'batch':
        norm_layer = functools.partial(eval('nn.BatchNorm{}'.format(dimension)))
    elif norm_type == 'instance':
        norm_layer = functools.partial(eval('nn.InstanceNorm{}'.format(dimension)))
    elif norm_type == 'group':
        norm_layer = functools.partial(eval('nn.GroupNorm{}'.format(dimension)))
    elif norm_type == 'none':
        norm_layer = lambda x: Identity()
    else:
        raise NotImplementedError('normalization layer [%s] is not found' % norm_type)
    return norm_layer


class UnetGenerator(nn.Module):
    """Create a Unet-based generator"""

    def __init__(self, input_channels,
                 output_channels,
                 downsampling_number,
                 filter_number_last_conv_layer=64,
                 norm_layer='batch',
                 dimension='2d'):
        super(UnetGenerator, self).__init__()
        # construct unet structure
        norm_layer = get_norm_layer(norm_layer, dimension)

        unet_block = UnetSkipConnectionBlock(filter_number_last_conv_layer * 8,
                                             filter_number_last_conv_layer * 8,
                                             dimension,
                                             input_channels=None,
                                             submodule=None,
                                             norm_layer=norm_layer,
                                             innermost=True)  # add the innermost layer
        for i in range(downsampling_number - 5):  # add intermediate layers with ngf * 8 filters
            unet_block = UnetSkipConnectionBlock(filter_number_last_conv_layer * 8,
                                                 filter_number_last_conv_layer * 8,
                                                 dimension,
                                                 input_channels=None,
                                                 submodule=unet_block,
                                                 norm_layer=norm_layer,
                                                 )
        # gradually reduce the number of filters from ngf * 8 to ngf
        unet_block = UnetSkipConnectionBlock(filter_number_last_conv_layer * 4,
                                             filter_number_last_conv_layer * 8,
                                             dimension,
                                             input_channels=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(filter_number_last_conv_layer * 2,
                                             filter_number_last_conv_layer * 4,
                                             dimension,
                                             input_channels=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        unet_block = UnetSkipConnectionBlock(filter_number_last_conv_layer,
                                             filter_number_last_conv_layer * 2,
                                             dimension,
                                             input_channels=None,
                                             submodule=unet_block,
                                             norm_layer=norm_layer)
        self.model = UnetSkipConnectionBlock(output_channels,
                                             filter_number_last_conv_layer,
                                             dimension,
                                             input_channels=input_channels,
                                             submodule=unet_block,
                                             outermost=True,
                                             norm_layer=norm_layer)  # add the outermost layer

    def forward(self, input):
        return self.model(input)


class UnetSkipConnectionBlock(nn.Module):
    """Defines the Unet submodule with skip connection.
        X -------------------identity----------------------
        |-- downsampling -- |submodule| -- upsampling --|
    """

    def __init__(self, outer_channels,
                 inner_channels,
                 dimension='2d',
                 input_channels=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.BatchNorm2d):
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost

        is_bias = use_bias(norm_layer)

        if input_channels is None:
            input_channels = outer_channels

        conv_layer, upsampling_mode = net_dimension(dimension)
        conv_layer_no_norm, _ = net_dimension(dimension)

        downconv = conv_layer(input_channels, inner_channels, kernel_size=3, stride=2, padding=1, bias=is_bias)
        downconv_no_norm = conv_layer_no_norm(input_channels, inner_channels, kernel_size=3, stride=2, padding=1, bias=True)
        downrelu = nn.LeakyReLU(0.2, True)
        downnorm = norm_layer(inner_channels)
        uprelu = nn.ReLU(True)
        upnorm = norm_layer(outer_channels)

        if outermost:
            upconv = nn.Sequential(*[nn.Upsample(scale_factor=2.0, mode=upsampling_mode, align_corners=True),
                                     conv_layer_no_norm(inner_channels * 2, outer_channels, kernel_size=3, stride=1, padding=1,
                                                bias=True)])
            down = [downconv_no_norm]
            up = [uprelu, upconv, nn.ReLU(True)]
            model = down + [submodule] + up
        elif innermost:
            upconv = nn.Sequential(*[nn.Upsample(scale_factor=2.0, mode=upsampling_mode, align_corners=True),
                                     conv_layer(inner_channels, outer_channels, kernel_size=3, stride=1, padding=1,
                                                bias=is_bias)])
            down = [downrelu, downconv_no_norm]
            up = [uprelu, upconv, upnorm]
            model = down + up
        else:
            upconv = nn.Sequential(*[nn.Upsample(scale_factor=2.0, mode=upsampling_mode, align_corners=True),
                                     conv_layer(inner_channels * 2, outer_channels, kernel_size=3, stride=1, padding=1,
                                                bias=is_bias)])
            down = [downrelu, downconv, downnorm]
            up = [uprelu, upconv, upnorm]

            model = down + [submodule] + up

        self.model = nn.Sequential(*model)

    def forward(self, x):
        if self.outermost:
            return self.model(x)
        else:  # add skip connections
            return torch.cat([x, self.model(x)], 1)


if __name__ == '__main__':
    print('Congrats! May the force be with you ...')
