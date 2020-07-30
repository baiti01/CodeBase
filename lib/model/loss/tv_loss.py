#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MIAI Lab | UT Southwestern Medical Center
# DATETIME: 2/13/2020 11:50 AM

# torch
import torch


def tv_loss(img, tv_weight):
    """
    Compute total variation loss.
    Inputs:
    - img: PyTorch Variable of shape (1, 3, H, W) holding an input image.
    - tv_weight: Scalar giving the weight w_t to use for the TV loss.
    Returns:
    - loss: PyTorch Variable holding a scalar giving the total variation loss
      for img weighted by tv_weight.
    """
    w_variance = torch.sum(torch.abs(img[:, :, :, :-1] - img[:, :, :, 1:]))
    h_variance = torch.sum(torch.abs(img[:, :, :-1, :] - img[:, :, 1:, :]))
    loss = tv_weight * (h_variance + w_variance)
    return loss
