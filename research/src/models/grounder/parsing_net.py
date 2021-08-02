# -*- coding: utf-8 -*-
"""Referring relationships grounder."""

import torch
from pdb import set_trace
from torch import nn
from scipy.stats import multivariate_normal
import numpy as np

from research.src.train_testers import GRNDTrainTester
from common.models.grounder import ParsingNet


class TrainTester(GRNDTrainTester):
    """Extends SGGTrainTester."""

    def __init__(self, net, config, features, obj_classifier, teacher):
        """Initialize instance."""
        super().__init__(net, config, features)
        self.mse = nn.MSELoss()
        self.bce = nn.BCELoss(reduction='none')

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs
        outputs = self._net_forward(batch, step)
        (
            subj_hmap, obj_hmap, subj_att, obj_att,
            nrm_subj_box_pred, nrm_obj_box_pred
        ) = outputs
        eps = 1e-8
        subj_hmap = torch.clamp(subj_hmap.flatten(1), eps, 1 - eps)
        obj_hmap = torch.clamp(obj_hmap.flatten(1), eps, 1 - eps)
        subj_att = torch.clamp(subj_att.flatten(1), eps, 1 - eps)
        obj_att = torch.clamp(obj_att.flatten(1), eps, 1 - eps)
        # Get Masks
        pairs = self.data_loader.get('pairs', batch, step)
        boxes = self.data_loader.get('object_rois', batch, step)
        masks = self.net.get_obj_masks(boxes)
        merged_masks = self.language_mask_merge(
            masks, batch['object_ids'][step])
        # subj_masks = masks[pairs[:, 0]].flatten(1)
        # obj_masks = masks[pairs[:, 1]].flatten(1)
        subj_merged_masks = merged_masks[pairs[:, 0]].flatten(1)
        obj_merged_masks = merged_masks[pairs[:, 1]].flatten(1)
        # Get normalized box Width, Height
        nrm_boxes = self.data_loader.get('object_rois_norm', batch, step)
        nrm_subj_boxes = nrm_boxes[pairs[:, 0]]
        nrm_obj_boxes = nrm_boxes[pairs[:, 1]]
        subj_gauss_masks = self.get_gauss_kernels(nrm_subj_boxes).flatten(1)
        obj_gauss_masks = self.get_gauss_kernels(nrm_obj_boxes).flatten(1)

        # Losses
        losses = {
            'BCE-subj': self.bce(subj_hmap, subj_gauss_masks).mean(-1),
            'BCE-obj': self.bce(obj_hmap, obj_gauss_masks).mean(-1),
            'BCE-subj_lang': self.bce(subj_att, subj_merged_masks).mean(-1),
            'BCE-obj_lang': self.bce(obj_att, obj_merged_masks).mean(-1),
            'MSE-subj': self.mse(nrm_subj_box_pred, nrm_subj_boxes[:, 2:]),
            'MSE-obj': self.mse(nrm_obj_box_pred, nrm_obj_boxes[:, 2:]),
        }
        # switch = 0.0 if self._epoch < 2 else 1.0
        loss = (
            losses['BCE-subj']
            + losses['BCE-obj']
            + losses['BCE-subj_lang']
            + losses['BCE-obj_lang']
            + losses['MSE-subj']
            + losses['MSE-obj']
        )
        return loss, losses

    def language_mask_merge(self, masks, ids):
        merged_masks = torch.stack(
            [torch.clamp(masks[ids == id].sum(0), max=1)
             for id in ids])
        return merged_masks

    def get_gauss_kernels(self, norm_boxes):
        """get gaussian kernels for gt box centers"""
        norm_boxes = norm_boxes.clone()
        norm_boxes *= self.net._mask_size
        x, y = np.mgrid[0:self.net._mask_size:1, 0:self.net._mask_size:1]
        pos = np.dstack((x, y))
        rv = [multivariate_normal([norm_boxes[i, 1], norm_boxes[i, 0]],
                                  [[0.5 * norm_boxes[i, 3], 0],
                                   [0, 0.5 * norm_boxes[i, 2]]]).pdf(pos)
              for i in range(len(norm_boxes))]
        # Normalize for bce loss
        gauss_masks = torch.stack(
            [torch.tensor(d/d.max()) for d in rv], dim=0
        ).to(self._device).type(torch.float)
        return gauss_masks


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a model."""
    net = ParsingNet(config)
    train_tester = TrainTester(net, config,
                               {'images'}, obj_classifier, teacher)
    train_tester.train_test()
