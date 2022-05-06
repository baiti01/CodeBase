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

# project
from lib.model.module.modules import pixel_shuffle, pixel_unshuffle


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
    def __init__(self,
                 input_channels,
                 output_channels,
                 downsampling_number,
                 filter_number_last_conv_layer=64,
                 norm_layer='instance',
                 dimension='3d'):
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
        input = pixel_unshuffle(input)
        input = self.model(input)
        input = pixel_shuffle(input)
        return input


def conv_layer_separable(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True):
    modules = []
    modules.append(nn.Conv3d(in_channels=in_channels,
                             out_channels=out_channels,
                             kernel_size=(1, kernel_size, kernel_size),
                             stride=(1, stride, stride),
                             padding=(0, padding, padding),
                             bias=bias))
    modules.append(nn.ReLU())
    modules.append(nn.Conv3d(in_channels=out_channels,
                             out_channels=out_channels,
                             kernel_size=(kernel_size, 1, 1),
                             stride=(stride, 1, 1),
                             padding=(padding, 0, 0),
                             bias=bias))
    #modules.append(nn.ReLU(inplace=True))
    return nn.Sequential(*modules)


class UnetSkipConnectionBlock(nn.Module):
    def __init__(self,
                 outer_channels,
                 inner_channels,
                 dimension='3d',
                 input_channels=None,
                 submodule=None,
                 outermost=False,
                 innermost=False,
                 norm_layer=nn.InstanceNorm3d):
        super(UnetSkipConnectionBlock, self).__init__()

        self.outermost = outermost

        is_bias = use_bias(norm_layer)

        if input_channels is None:
            input_channels = outer_channels

        _, upsampling_mode = net_dimension(dimension)

        # ###
        conv_layer = conv_layer_separable
        conv_layer_no_norm = conv_layer_separable

        downconv = conv_layer(input_channels, inner_channels, kernel_size=3, stride=2, padding=1, bias=is_bias)
        downconv_no_norm = conv_layer_no_norm(input_channels, inner_channels, kernel_size=3, stride=2, padding=1, bias=True)
        downrelu = nn.ReLU(True)
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
    import time
    model = UnetGenerator(input_channels=8, output_channels=8, downsampling_number=5, dimension='3d')
    print(model)
    model = model.to('cuda')
    input = torch.randn((1, 1, 64, 256, 256)).to('cuda')
    for i in range(10):
        output = model(input)

    t1 = time.time()
    for i in range(50):
        output = model(input)
    t2 = time.time()
    print('Average time per one forward: {}'.format((t2 - t1)/50))
    print('Congrats! May the force be with you ...')
