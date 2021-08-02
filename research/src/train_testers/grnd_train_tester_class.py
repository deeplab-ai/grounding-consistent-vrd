# -*- coding: utf-8 -*-
"""
A class for training/testing a network Grounding.

Methods _compute_loss and _net_outputs assume that _net_forward
returns (subj_masks, obj_masks, ...).
They should be re-implemented if that's not the case.
"""

import numpy as np
import torch
from torch import nn
from tqdm import tqdm
import random

from research.src.evaluators import (
    GroundEvaluator
)
from research.src.data_loaders import SGGDataset, SGGDataLoader

from .base_train_tester_class import BaseTrainTester


class GRNDTrainTester(BaseTrainTester):
    """
    Train and test utilities for Grounding.

    Inputs upon initialization:
        - net: PyTorch nn.Module, the network to train/test
        - config: class Config, see config.py
        - features: set of str, features to load for net
        - obj_classifier: ObjectClassifier object (see corr. API)
        - teacher: a loaded SGG model
    """

    def __init__(self, net, config, features):
        """Initiliaze train/test instance."""
        super().__init__(net, config, features)
        self.bce = nn.BCELoss(reduction='none')

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        # Settings and loading
        self.logger.info(
            "Test %s on %s on %s" % (self._net_name, self._task, self._dataset)
        )
        self.training_mode = False
        self.net.eval()
        self.net.to(self._device)
        self._set_data_loaders(mode_ids={'test': 2})
        self.data_loader = self._data_loaders['test']
        grnd_eval = GroundEvaluator(self.annotation_loader)

        # Forward pass on test set
        for batch in tqdm(self.data_loader):
            for step in range(len(batch['filenames'])):
                # Get estimated masks and b-box norm dimentions
                (
                    subj_masks, obj_masks, nrm_subj_box, nrm_obj_box
                ) = self._net_outputs(batch, step)
                mask_shape = (
                    subj_masks.shape if subj_masks is not None
                    else obj_masks.shape
                )
                # Get gt masks
                boxes = self.data_loader.get('object_rois', batch, step)
                masks = self.net.get_obj_masks(boxes, mask_size=mask_shape[-1])
                pairs = self.data_loader.get('pairs', batch, step)
                # Evaluation step
                grnd_eval.step(
                    batch['filenames'][step],
                    [subj_masks, obj_masks],
                    [nrm_subj_box, nrm_obj_box],
                    [masks[pairs[:, 0]], masks[pairs[:, 0]]]
                )
        # Print metrics
        grnd_eval.print_stats()

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs
        outputs = self._net_forward(batch, step)
        eps = 1e-8
        subj_att = torch.clamp(outputs[0].flatten(1), eps, 1 - eps)
        obj_att = torch.clamp(outputs[1].flatten(1), eps, 1 - eps)
        # Targets
        pairs = self.data_loader.get('pairs', batch, step)
        boxes = self.data_loader.get('object_rois', batch, step)
        masks = self.net.get_obj_masks(boxes, mask_size=outputs[0].shape[-1])
        subj_masks = masks[pairs[:, 0]].flatten(1)
        obj_masks = masks[pairs[:, 1]].flatten(1)
        # Losses
        losses = {
            'BCE-subj': self.bce(subj_att, subj_masks).mean(-1),
            'BCE-obj': self.bce(obj_att, obj_masks).mean(-1)
        }
        loss = losses['BCE-subj'] + losses['BCE-obj']
        return loss, losses

    def _net_forward(self, batch, step):
        """Return a tuple of scene graph generator's outputs."""
        return self.net(
            self.data_loader.get('images', batch, step),
            self.data_loader.get('object_rois', batch, step),
            self.data_loader.get('predicate_ids', batch, step),
            self.data_loader.get('object_ids', batch, step),
            self.data_loader.get('pairs', batch, step),
            self.data_loader.get('image_info', batch, step)
        )

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch."""
        rest_outputs = self._net_forward(batch, step)
        subj_masks, obj_masks = rest_outputs[:2]
        nrm_subj_box, nrm_obj_box = rest_outputs[4], rest_outputs[5]
        return subj_masks, obj_masks, nrm_subj_box, nrm_obj_box

    def _set_data_loaders(self, mode_ids={'train': 0, 'val': 1, 'test': 2}):
        annotations = np.array(self.annotation_loader.get_annos())
        split_ids = np.array([anno['split_id'] for anno in annotations])
        datasets = {
            split: SGGDataset(
                annotations[split_ids == split_id].tolist(),
                self.config, self.features)
            for split, split_id in mode_ids.items()
        }
        self._data_loaders = {
            split: SGGDataLoader(
                datasets[split], batch_size=self._batch_size,
                shuffle=split == 'train', num_workers=self._num_workers,
                drop_last=split != 'test', device=self._device)
            for split in mode_ids
        }
        if 'overfit' in mode_ids:
            random.seed(0)
            overfit_idx = random.sample(list(np.where(split_ids == 0)[0]), 20)
            datasets['overfit'] = SGGDataset(
                annotations[overfit_idx].tolist(),
                self.config, self.features)
            self._data_loaders['overfit'] = SGGDataLoader(
                datasets['overfit'], batch_size=1,
                shuffle=True, num_workers=self._num_workers,
                drop_last=False, device=self._device)


def nonepy(tensor):
    """Convert tensor to numpy but check for Nones."""
    return tensor.cpu().numpy() if tensor is not None else None
