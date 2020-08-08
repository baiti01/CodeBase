#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 7/31/2020 11:23 AM

# torch
import torch
import torch.nn as nn

# project
from lib.model.utils.modules import conv_bn_relu


class PlainNet(nn.Module):
    def __init__(self, input_channels=1, output_channels=1, layer_number=8, channel_number=32, is_3d=False,
                 norm_type='instance'):
        super(PlainNet, self).__init__()

        self.stem = conv_bn_relu(in_channels=input_channels, out_channels=channel_number,
                                 norm_type=norm_type, is_3d=False)

        module_list = []
        for i in range(layer_number):
            current_module = conv_bn_relu(in_channels=channel_number,
                                          out_channels=channel_number,
                                          is_3d=is_3d,
                                          norm_type=norm_type)
            module_list.append(current_module)
        self.feature = nn.Sequential(*module_list)

        self.predictor = nn.Sequential(*[nn.Conv2d(in_channels=channel_number, out_channels=output_channels, kernel_size=1),
                                         nn.ReLU()])

    def forward(self, input):
        input = self.stem(input)
        input = self.feature(input)
        output = self.predictor(input)
        return output


if __name__ == '__main__':
    model = PlainNet().cuda()
    print(model)
    input = torch.randn((1, 1, 512, 512)).cuda()
    output = model(input)
    print('Congrats! May the force be with you ...')
