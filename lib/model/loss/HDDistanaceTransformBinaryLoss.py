#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 12/12/2020 12:04 PM

# sys
import numpy as np
from scipy.ndimage import distance_transform_edt

# torch
import torch
import torch.nn as nn


def softmax_helper(x):
    # copy from: https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunet/utilities/nd_softmax.py
    rpt = [1 for _ in range(len(x.size()))]
    rpt[1] = x.size(1)
    x_max = x.max(1, keepdim=True)[0].repeat(*rpt)
    e_x = torch.exp(x - x_max)
    return e_x / e_x.sum(1, keepdim=True).repeat(*rpt)


def compute_edts_forhdloss(segmentation):
    res = np.zeros(segmentation.shape)
    for i in range(segmentation.shape[0]):
        posmask = segmentation[i]
        negmask = ~posmask
        res[i] = distance_transform_edt(posmask) + distance_transform_edt(negmask)
    return res


class HDDTBinaryLoss(nn.Module):
    def __init__(self):
        """
        compute Hausdorff loss for binary segmentation
        https://arxiv.org/pdf/1904.10030v1.pdf
        """
        super(HDDTBinaryLoss, self).__init__()

    def forward(self, net_output, target):
        """
        net_output: (batch_size, 2, x,y,z)
        target: ground truth, shape: (batch_size, 1, x,y,z)
        """
        #net_output = softmax_helper(net_output)
        net_output = torch.sigmoid(net_output)
        #pc = net_output[:, 1, ...].type(torch.float32)
        pc = net_output[:, 0, ...].type(torch.float32)
        gt = target[:, 0, ...].type(torch.float32)
        with torch.no_grad():
            pc_dist = compute_edts_forhdloss(pc.cpu().numpy() > 0.5)
            gt_dist = compute_edts_forhdloss(gt.cpu().numpy() > 0.5)
        # print('pc_dist.shape: ', pc_dist.shape)

        pred_error = (gt - pc) ** 2
        dist = pc_dist ** 2 + gt_dist ** 2  # \alpha=2 in eq(8)

        dist = torch.from_numpy(dist)
        if dist.device != pred_error.device:
            dist = dist.to(pred_error.device).type(torch.float32)

        #multipled = torch.einsum("bxyz,bxyz->bxyz", pred_error, dist)
        multipled = torch.einsum("bxy,bxy->bxy", pred_error, dist)
        hd_loss = multipled.mean()

        return hd_loss


if __name__ == '__main__':
    test_data = np.load(r'D:\TIBAI\project\Segmentation\InteractiveSegmentation\test.npz')
    gt = test_data['gt']
    pred = test_data['pred']
    output = test_data['output']
    gt_tensor = torch.from_numpy(gt).unsqueeze(0).unsqueeze(0)
    output_tensor = torch.from_numpy(output).unsqueeze(0).unsqueeze(0)

    criterion = HDDTBinaryLoss()

    loss = criterion(output_tensor, gt_tensor)
    print('Congrats! May the force be with you ...')