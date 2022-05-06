#!/usr/bin/env python
# -*- coding:utf-8 -*-
# AUTHOR: Ti Bai
# EMAIL: tibaiw@gmail.com
# AFFILIATION: MAIA Lab | UT Southwestern Medical Center
# DATETIME: 11/13/2020 7:29 AM

# sys
import numpy as np
from scipy.ndimage.morphology import distance_transform_edt, binary_erosion, generate_binary_structure
from collections import namedtuple
from scipy.special import softmax
from scipy.ndimage import _ni_support
import logging

# torch
import torch

logger = logging.getLogger(__name__)
Click = namedtuple('Click', ['is_positive', 'is_boundary', 'coords'])


class Clicker(object):
    def __init__(self, gt_mask=None, click_type='boundary', phase='train'):
        super(Clicker, self).__init__()
        assert click_type == 'boundary' or click_type == 'false'
        self.gt_mask = gt_mask
        self.click_type = click_type
        self._reset_clicks()
        self.phase = phase

    def make_next_click(self, pred_mask, slice_idx):
        assert self.gt_mask is not None
        self.current_gt_mask = self.gt_mask[slice_idx]
        self.slice_idx = slice_idx

        if self.click_type == 'boundary':
            click = self._get_boundary_click(pred_mask=pred_mask)
        elif self.click_type == 'false':
            click = self._get_false_click(pred_mask=pred_mask)
        else:
            raise ValueError(f'Unsupported click type: {self.click_type}')

        self.add_click(click)

    def _get_boundary_click(self, pred_mask):
        if 0 == np.count_nonzero(self.current_gt_mask):
            raise RuntimeError('GT Mask does not contain any binary object.')

        if 0 == np.count_nonzero(pred_mask):
            footprint = generate_binary_structure(self.current_gt_mask.ndim, 1)
            mask_border = self.current_gt_mask.astype(np.bool) ^ binary_erosion(self.current_gt_mask.astype(np.bool),
                                                                                structure=footprint, iterations=1)
            coords_max_distance = np.argwhere(mask_border)[0]

            # add slice index
            coords_max_distance = np.insert(coords_max_distance, 0, self.slice_idx)
        else:
            current_surface_distance, current_coords = surface_distances(self.current_gt_mask, pred_mask)

            # add slice index
            current_coords = np.concatenate((np.ones((current_coords.shape[0], 1), dtype=current_coords.dtype) * self.slice_idx, current_coords), axis=1)

            # remove those clicked point
            clicked_points = [list(x.coords) for x in self.clicks_list]
            for idx, current_point in enumerate(list(current_coords)):
                if list(current_point) in clicked_points:
                    current_surface_distance[idx] = 0

            if self.phase.upper() == 'train'.upper():
                # probabilistic sampling for training phase
                random_index = np.random.choice(np.arange(len(current_surface_distance)),
                                                p=softmax(current_surface_distance))
                coords_max_distance = current_coords[random_index]
            else:
                # max distance based sampling for testing phase
                coords_max_distance = current_coords[np.argmax(current_surface_distance)]
        return Click(is_positive=True, is_boundary=True, coords=coords_max_distance)

    def _get_false_click(self, pred_mask, padding=True):
        # todo: need to test the 3D case, i.e., add the slice index
        fn_mask = np.logical_and(self.gt_mask, np.logical_not(pred_mask))
        fp_mask = np.logical_and(np.logical_not(self.gt_mask), pred_mask)

        if padding:
            fn_mask = np.pad(fn_mask, ((1, 1), (1, 1)), 'constant')
            fp_mask = np.pad(fp_mask, ((1, 1), (1, 1)), 'constant')

        fn_mask_distance_transform = distance_transform_edt(fn_mask)
        fp_mask_distance_transform = distance_transform_edt(fp_mask)

        if padding:
            fn_mask_distance_transform = fn_mask_distance_transform[1:-1, 1:-1]
            fp_mask_distance_transform = fp_mask_distance_transform[1:-1, 1:-1]

        fn_mask_distance_transform = fn_mask_distance_transform * self._not_clicked_map
        fp_mask_distance_transform = fp_mask_distance_transform * self._not_clicked_map
        if self.phase == 'train':
            fn_coord = np.argwhere(fn_mask_distance_transform)
            fp_coord = np.argwhere(fp_mask_distance_transform)
            false_coords = list(fn_coord) + list(fp_coord)

            fn_value = [fn_mask_distance_transform[x[0], x[1]] for x in fn_coord]
            fp_value = [fp_mask_distance_transform[x[0], x[1]] for x in fp_coord]
            false_values = np.array(fn_value + fp_value)

            random_index = np.random.choice(np.arange(len(false_values)), p=softmax(false_values))
            coords_y, coords_x = false_coords[random_index]
            is_positive = True if random_index < len(fn_value) else False
            return Click(is_positive=is_positive, is_boundary=False, coords=(self.slice_idx, coords_y, coords_x))
        else:
            fn_max_distance = np.max(fn_mask_distance_transform)
            fp_max_distance = np.max(fp_mask_distance_transform)

            is_positive = (fn_max_distance > fp_max_distance)
            if is_positive:
                coords_y, coords_x = np.where(fn_mask_distance_transform == fn_max_distance)
            else:
                coords_y, coords_x = np.where(fp_mask_distance_transform == fp_max_distance)

            return Click(is_positive=is_positive, is_boundary=False, coords=(self.slice_idx, coords_y[0], coords_x[0]))

    def get_clicks(self, clicks_limit=None):
        return self.clicks_list[:clicks_limit]

    def get_points(self, clicks_list):
        if self.click_type == 'boundary':
            boundary_clicks = [current_click.coords for current_click in clicks_list if current_click.is_boundary]
            total_clicks = [boundary_clicks]
        elif self.click_type == 'false':
            total_clicks = []

            num_boundary_clicks = sum([current_click.is_boundary for current_click in clicks_list])

            num_positive_clicks = sum([current_click.is_positive and (not current_click.is_boundary)
                                       for current_click in clicks_list])
            num_negative_clicks = len(clicks_list) - num_positive_clicks - num_boundary_clicks
            num_max_points = max(num_boundary_clicks, num_positive_clicks, num_negative_clicks)

            positive_clicks = [current_click.coords for current_click in clicks_list
                               if current_click.is_positive and (not current_click.is_boundary)]
            positive_clicks = positive_clicks + (num_max_points - num_positive_clicks) * [(-1, -1)]

            negative_clicks = [current_click.coords for current_click in clicks_list
                               if (not current_click.is_positive) and (not current_click.is_boundary)]
            negative_clicks = negative_clicks + (num_max_points - num_negative_clicks) * [(-1, -1)]

            total_clicks.append(positive_clicks + negative_clicks)
        else:
            raise ValueError(f'Unsupported click type: {self.click_type}')

        total_clicks = torch.Tensor(total_clicks)
        if torch.cuda.is_available():
            total_clicks = total_clicks.cuda()

        return total_clicks

    def add_click(self, click):
        coords = click.coords
        if click.is_boundary:
            self.num_boundary_clicks += 1
        else:
            if click.is_positive:
                self.num_positive_clicks += 1
            else:
                self.num_negative_clicks += 1

        self.clicks_list.append(click)

        if self.gt_mask is not None:
            self._not_clicked_map[coords[0], coords[1], coords[2]] = False

    def _reset_clicks(self):
        if self.gt_mask is not None:
            self._not_clicked_map = np.ones_like(self.gt_mask, dtype=np.bool)
        self.num_positive_clicks = 0
        self.num_negative_clicks = 0
        self.num_boundary_clicks = 0

        self.clicks_list = []

    def __len__(self):
        return len(self.clicks_list)


def surface_distances(result, reference, voxelspacing=None, connectivity=1):
    """
    The distances between the surface voxel of binary objects in result and their
    nearest partner surface voxel of a binary object in reference.
    """
    result = np.atleast_1d(result.astype(np.bool))
    reference = np.atleast_1d(reference.astype(np.bool))
    if voxelspacing is not None:
        voxelspacing = _ni_support._normalize_sequence(voxelspacing, result.ndim)
        voxelspacing = np.asarray(voxelspacing, dtype=np.float64)
        if not voxelspacing.flags.contiguous:
            voxelspacing = voxelspacing.copy()

    # binary structure
    footprint = generate_binary_structure(result.ndim, connectivity)

    # test for emptiness
    if 0 == np.count_nonzero(result):
        raise RuntimeError('The first supplied array does not contain any binary object.')
    if 0 == np.count_nonzero(reference):
        raise RuntimeError('The second supplied array does not contain any binary object.')

        # extract only 1-pixel border line of objects
    result_border = result ^ binary_erosion(result, structure=footprint, iterations=1)
    reference_border = reference ^ binary_erosion(reference, structure=footprint, iterations=1)

    # compute average surface distance
    # Note: scipys distance transform is calculated only inside the borders of the
    #       foreground objects, therefore the input has to be reversed
    dt = distance_transform_edt(~reference_border, sampling=voxelspacing)
    sds = dt[result_border]
    coords = np.argwhere(result_border)
    return sds, coords
