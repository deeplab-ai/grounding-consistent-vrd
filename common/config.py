# -*- coding: utf-8 -*-
"""Configuration parameters for each dataset and task."""

from collections import Counter
import json
from os import path as osp

import numpy as np
import torch


class Config:
    """
    A class to configure global or dataset/task-specific parameters.

    Inputs:
        Dataset/task:
            - dataset: str, dataset codename, e.g. 'VRD', 'VG200' etc.
            - task: str, task codename, supported are:
                - 'preddet': Predicate Detection
                - 'predcls': Predicate Classification
                - 'sgcls': Scene Graph Classification
                - 'sggen': Scene Graph Generation
                - 'objcls': Object Classification
                - 'objdet': Object Detection
        Data handling params:
            - bg_perc: float in [0, 1], perc. of background annotations
            - filter_duplicate_rels: bool, filter relations
                annotated more than once (during training)
            - filter_multiple_preds: bool, sample a single
                predicate per object pair (during training)
            - max_train_samples: int or None, keep classes with samples
                less than this number
            - num_tail_classes: int or None, keep the num_tail_classes
                with the fewest samples
            - test_on_negatives: flag, weather to test and report precision
        General:
            - device: str, device (gpu/cpu), e.g. 'cuda:0'
            - prerequisites_path: str, path where data are stored
            - rel_batch_size: int, split relations in such sub-batches
            - use_coco: bool, (for demo) use coco objects
    """

    def __init__(self, dataset='VRD', task='preddet', bg_perc=None,
                 filter_duplicate_rels=False, filter_multiple_preds=False,
                 max_train_samples=None, num_tail_classes=None,
                 prerequisites_path='prerequisites/', rel_batch_size=64,
                 use_coco=False, device='cuda:0',
                 test_on_negatives=False, **kwargs):
        """Initialize configuration instance."""
        self.dataset = dataset
        self.task = task
        self._bg_perc = bg_perc
        self.filter_duplicate_rels = filter_duplicate_rels
        self.filter_multiple_preds = filter_multiple_preds
        self.max_train_samples = max_train_samples
        self.num_tail_classes = num_tail_classes
        self.prerequisites_path = prerequisites_path
        self.rel_batch_size = rel_batch_size
        self.use_coco = use_coco
        self._device = device
        self.test_on_negatives = test_on_negatives
        self._json_path = osp.join(prerequisites_path, 'sgg_annos/', '')
        self._set_dataset_classes(dataset)

    def _set_dataset_classes(self, dataset):
        """Load dataset classes."""
        # Object classes
        self.obj_classes = None
        obj_json = osp.join(self._json_path, dataset + '_objects.json')
        if osp.exists(self._json_path) and osp.exists(obj_json):
            with open(obj_json) as fid:
                self.obj_classes = json.load(fid)
                self.num_obj_classes = len(self.obj_classes)
        # Predicate classes
        self.rel_classes = None
        pred_json = osp.join(self._json_path, dataset + '_predicates.json')
        if osp.exists(self._json_path) and osp.exists(pred_json):
            with open(pred_json) as fid:
                self.rel_classes = json.load(fid)
                self.num_rel_classes = len(self.rel_classes)
        # Filter classes
        keep = None
        if self.max_train_samples or self.num_tail_classes:
            preddet_json = osp.join(self._json_path, dataset + '_preddet.json')
            with open(preddet_json) as fid:
                annotations = json.load(fid)
                pred_list = [
                    pred for anno in annotations
                    for pred in anno['relations']['ids']
                    if anno['split_id'] == 0
                ]
                pred_counter = Counter(pred_list)
            if self.max_train_samples is not None:
                keep = np.sort([
                    pred for pred, cnt in pred_counter.items()
                    if cnt < self.max_train_samples
                ])
            elif self.num_tail_classes is not None:
                keep = np.array([
                    pred_counter[pred]
                    for pred in range(self.num_rel_classes - 1)
                ])
                keep = np.argsort(keep)[:self.num_tail_classes]
        self.classes_to_keep = keep

    @property
    def bg_perc(self):
        """Return percentage of background annotations."""
        if self._bg_perc is None and self.task == 'preddet':
            return 0.0
        if self._bg_perc is None:
            return 1.0
        return self._bg_perc

    @property
    def device(self):
        """Return device checking whether to use CUDA or not."""
        if torch.cuda.is_available():
            return self._device
        return 'cpu'

    @property
    def num_classes(self):
        """Return number of classes depending on task."""
        if self.task in {'objcls', 'objdet'}:
            return self.num_obj_classes
        return self.num_rel_classes

    @property
    def orig_img_path(self):
        """Return path of stored dataset images."""
        _dataset = 'VG' if 'VG' in self.dataset else self.dataset
        return osp.join(self.prerequisites_path, _dataset, 'images', '')

    @property
    def paths(self):
        """Return a dict of paths useful to train/test/inference."""
        return {'json_path': self._json_path}

    @property
    def relations_per_img_limit(self):
        """Return upper limit of examined relations in a train image."""
        return 2000

    @property
    def train_top(self):
        """Return whether to retrain the layer before the classifier."""
        return self.dataset not in {'VRD', 'UnRel'}
