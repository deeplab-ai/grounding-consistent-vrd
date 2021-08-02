# -*- coding: utf-8 -*-
"""Functions for training and testing an object classifier."""

import json

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from research.src.data_loaders import ObjDataset, SGGDataLoader
from research.src.evaluators import ObjectClsEvaluator
from .base_train_tester_class import BaseTrainTester


class ObjClsTrainTester(BaseTrainTester):
    """
    Train and test utilities for Object Classification.

    Inputs upon initialization:
        - net: PyTorch nn.Module, the network to train/test
        - config: class Config, see config.py
        - features: set of str, features to load for net
        - teacher: a loaded Object Classifier model
    """

    def __init__(self, net, config, features, teacher=None):
        """Initiliaze train/test instance."""
        super().__init__(net, config, features)
        self.teacher = teacher

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        # Settings and loading
        self.logger.info(
            "Test %s on objcls on %s" % (self._net_name, self._dataset))
        self.net.eval()
        self.net.to(self._device)
        self._set_data_loaders(mode_ids={'test': 2})
        self.data_loader = self._data_loaders['test']
        evaluator = ObjectClsEvaluator(self.annotation_loader)

        # Forward pass on test set
        results = {}
        for batch in tqdm(self.data_loader):
            for step in range(len(batch['filenames'])):
                scores = self._net_outputs(batch, step).cpu().numpy()
                filename = batch['filenames'][step]
                evaluator.step(filename, scores)
                results.update({  # top-5 classification results
                    filename: {
                        'scores': np.sort(scores)[:, ::-1][:, :5].tolist(),
                        'classes': scores.argsort(1)[:, ::-1][:, :5].tolist()
                    }})
        # Print metrics and save results
        evaluator.print_stats()
        with open(self._results_path + 'results.json', 'w') as fid:
            json.dump(results, fid)

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        scores = self._net_forward(batch, step)
        targets = self.data_loader.get('object_ids', batch, step)
        losses = {'obj_loss': self.criterion(scores, targets)}
        loss = losses['obj_loss']
        if self.teacher is not None:
            losses['KD'] = self._kd_loss(scores, batch, step)
            if self.training_mode:
                loss += losses['KD']
        return loss, losses

    @torch.no_grad()
    def _kd_loss(self, scores, batch, step):
        """Compute knowledge distillation loss."""
        t_outputs = self.teacher(
            self.data_loader.get('images', batch, step),
            self.data_loader.get('object_rois', batch, step),
            self.data_loader.get('object_ids', batch, step),
            self.data_loader.get('image_info', batch, step)
        )
        return 50 * F.kl_div(
            F.log_softmax(scores / 5, 1), F.softmax(t_outputs / 5, 1),
            reduction='none'
        ).mean(1)

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch."""
        return self._net_forward(batch, step)

    def _net_forward(self, batch, step):
        """Support the forward pass of this network."""
        return self.net(
            self.data_loader.get('images', batch, step),
            self.data_loader.get('object_rois', batch, step),
            self.data_loader.get('object_ids', batch, step),
            self.data_loader.get('image_info', batch, step)
        )

    def _set_data_loaders(self, mode_ids={'train': 0, 'val': 1, 'test': 2}):
        annotations = np.array(self.annotation_loader.get_annos())
        split_ids = np.array([anno['split_id'] for anno in annotations])
        datasets = {
            split: ObjDataset(
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
