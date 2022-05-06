#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# DATETIME: 10/14/2019 4:16 PM
# FILE: modules.py

def pixel_unshuffle(x, upscale_factor=2):
    dimension = 2 if len(x.shape) == 4 else 3
    if dimension == 2:
        b, c, h, w = x.shape
        c_out, h_out, w_out = c * (upscale_factor ** dimension), h // upscale_factor, w // upscale_factor
        x = x.contiguous().view(b, c, h_out, upscale_factor, w_out, upscale_factor)
        x = x.permute(0, 1, 3, 5, 2, 4).contiguous().view(b, c_out, h_out, w_out)
    else:
        b, c, d, h, w = x.shape
        c_out, d_out, h_out, w_out = c * (
                    upscale_factor ** dimension), d // upscale_factor, h // upscale_factor, w // upscale_factor
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


if __name__ == '__main__':
    print('Congrats! May the force be with you ...')
