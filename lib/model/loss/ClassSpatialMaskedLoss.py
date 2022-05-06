#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 10/08/2021 7:02 PM

import inspect
import warnings
from typing import Callable, Optional, Union

import torch
from torch.nn.modules.loss import _Loss
from monai.losses import DiceLoss


class ClassSpatialMaskedLoss(_Loss):
    """
    This is a wrapper class for the loss functions.  It allows for additional
    weighting masks to be applied to both input and target.

    See Also:
        - :py:class:`monai.losses.MaskedDiceLoss`
    """

    def __init__(self, loss: Union[Callable, _Loss], *loss_args, **loss_kwargs) -> None:
        """
        Args:
            loss: loss function to be wrapped, this could be a loss class or an instance of a loss class.
            loss_args: arguments to the loss function's constructor if `loss` is a class.
            loss_kwargs: keyword arguments to the loss function's constructor if `loss` is a class.
        """
        super().__init__()
        self.loss = loss(*loss_args, **loss_kwargs) if inspect.isclass(loss) else loss
        if not callable(self.loss):
            raise ValueError("The loss function is not callable.")

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should be BNH[WD] or 1NH[WD].
        """
        if mask is None:
            warnings.warn("No mask value specified for the MaskedLoss.")
            return self.loss(input, target)

        if input.dim() != mask.dim():
            warnings.warn(f"Dim of input ({input.shape}) is different from mask ({mask.shape}).")
        if input.shape[0] != mask.shape[0] and mask.shape[0] != 1:
            raise ValueError(f"Batch size of mask ({mask.shape}) must be one or equal to input ({input.shape}).")
        if target.dim() > 1:
            if input.shape[1:] != mask.shape[1:]:
                warnings.warn(f"Spatial size and channel size of input ({input.shape}) is different from mask ({mask.shape}).")
        return self.loss(torch.sigmoid(input) * mask, target * mask)


class ClassSpatialMaskedDiceLoss(DiceLoss):
    """
    Add an additional `masking` process before `DiceLoss`, accept a binary mask ([0, 1]) indicating a region,
    `input` and `target` will be masked by the region: region with mask `1` will keep the original value,
    region with `0` mask will be converted to `0`. Then feed `input` and `target` to normal `DiceLoss` computation.
    This has the effect of ensuring only the masked region contributes to the loss computation and
    hence gradient calculation.

    """

    def __init__(self, *args, **kwargs) -> None:
        """
        Args follow :py:class:`monai.losses.DiceLoss`.
        """
        super().__init__(*args, **kwargs)
        self.spatial_weighted = ClassSpatialMaskedLoss(loss=super().forward)

    def forward(self, input: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None):
        """
        Args:
            input: the shape should be BNH[WD].
            target: the shape should be BNH[WD].
            mask: the shape should B1H[WD] or 1NH[WD].
        """
        return self.spatial_weighted(input=input, target=target, mask=mask)