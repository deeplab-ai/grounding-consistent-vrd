# -*- coding: utf-8 -*-
"""Configuration parameters for each dataset and task."""

import logging
from math import ceil
import os
from os import path as osp

from colorlog import ColoredFormatter

from common.config import Config


class ResearchConfig(Config):
    """
    A class to configure global or dataset/task-specific parameters.

    Inputs (see common.config.py for parent arguments):
        Dataset/task:
            - net_name: str, name of trained model
            - phrase_recall: bool, whether to evaluate phrase recall
            - test_dataset: str or None, dataset to evaluate
        Data handling params:
            - annotations_per_batch: int, number of desired annotations
                per batch on average, in terms of relations or objects
            - augment_annotations: bool, distort boxes to augment
        Evaluation params:
            - compute_accuracy: bool, measure accuracy, not recall
            - use_merged: bool, use merged annotations in evaluation
        Loss functions:
            - use_multi_tasking: bool, use multi-tasking to
                separately decide for object relevance
            - use_weighted_ce: bool, use weighted cross-entropy
        Training params:
            - batch_size: int or None, batch size in images (if custom)
            - epochs: int or None, number of training epochs
            - learning_rate: float, learning rate of classifier
            - weight_decay: float, weight decay of optimizer
        Learning rate policy:
            - apply_dynamic_lr: bool, adapt lr to preserve
                lr / annotations per batch
            - use_early_stopping: bool, lr policy with early stopping
            - restore_on_plateau: bool, whether to restore checkpoint
                on validation metric's plateaus (only effective in early
                stopping)
            - patience: int, number of epochs to consider a plateau
        General:
            - commit: str, commit name to tag model
            - num_workers: int, workers employed by the data loader
    """

    def __init__(self, net_name='', phrase_recall=False, test_dataset=None,
                 annotations_per_batch=128, augment_annotations=True,
                 compute_accuracy=False, use_merged=False,
                 use_multi_tasking=True,
                 use_weighted_ce=False, batch_size=None, epochs=None,
                 learning_rate=0.002, weight_decay=None,
                 apply_dynamic_lr=False, use_early_stopping=True,
                 restore_on_plateau=True, patience=1, commit='', num_workers=2,
                 use_consistency_loss=False, use_graphl_loss=False,
                 misc_params=None, **kwargs):
        """Initialize configuration instance.
        :param use_graphl_loss:
        """
        super().__init__(**kwargs)
        if misc_params is None:
            misc_params = {}
        self.net_name = '_'.join([
            net_name,
            (self.task if self.task not in {'sgcls', 'sggen'} else 'predcls'),
            self.dataset if self.dataset != 'UnRel' else 'VRD'
        ])
        self.phrase_recall = phrase_recall
        self.test_dataset = (
            self.dataset if test_dataset is None
            else test_dataset
        )
        self._annotations_per_batch = annotations_per_batch
        self.augment_annotations = augment_annotations
        self.use_multi_tasking = use_multi_tasking
        self.use_weighted_ce = use_weighted_ce
        self.use_consistency_loss = use_consistency_loss
        self.use_graphl_loss = use_graphl_loss
        self.compute_accuracy = compute_accuracy and self.task == 'preddet'
        self.use_merged = use_merged
        self._batch_size = batch_size
        self._epochs = epochs
        self.learning_rate = learning_rate
        self._weight_decay = weight_decay
        self.apply_dynamic_lr = apply_dynamic_lr
        self.use_early_stopping = use_early_stopping
        self.restore_on_plateau = restore_on_plateau
        self.patience = patience
        self.commit = (
            commit + '_' + self.net_name if commit != ''
            else self.net_name
        )
        self.num_workers = num_workers
        self._set_dataset_task_annos_per_img()
        self._set_logger()

    def reset(self, custom_dataset=None):
        """Reset instance to handle another dataset."""
        self.dataset = (
            self.test_dataset if custom_dataset is None
            else custom_dataset
        )
        self._set_dataset_classes(self.dataset)

    def _set_dataset_task_annos_per_img(self):
        """
        Different number of image-wise annotations per dataset-task.

        All fields except for 'objects' refer to predicate annotations:
            - If duplicates_filtered, clear relations annotated > 1 time
            - If predicates_filtered, sample a single predicate per pair
            - If pairs, use all possible pairs of objects
        """
        self._annos_per_img = {
            'VG200': {
                'relations': 6.98,
                'duplicates_filtered': 4.69,
                'predicates_filtered': 4.45,
                'objects': 10.87,
                'pairs': 146.3,
                'max_objects': 45,
            },
            'VG80K': {
                'relations': 21.96,
                'duplicates_filtered': 18.89,
                'predicates_filtered': 18.1,
                'objects': 23.48,
                'pairs': 696.85,
                'max_objects': 25
            },
            'VGMSDN': {
                'relations': 11.02,
                'duplicates_filtered': 9.13,
                'predicates_filtered': 8.79,
                'objects': 12.48,
                'pairs': 190.05,
                'max_objects': 83
            },
            'VGVTE': {
                'relations': 10.94,
                'duplicates_filtered': 9.28,
                'predicates_filtered': 9.03,
                'objects': 13.04,
                'pairs': 243.76,
                'max_objects': 110
            },
            'VRD': {
                'relations': 8.02,
                'duplicates_filtered': 7.89,
                'predicates_filtered': 7.13,
                'objects': 7,
                'pairs': 52.98,
                'max_objects': 21
            },
            'VrR-VG': {
                'relations': 3.45,
                'duplicates_filtered': 3.03,
                'predicates_filtered': 2.97,
                'objects': 4.79,
                'pairs': 34.63,
                'max_objects': 64
            },
            'sVG': {
                'relations': 10.89,
                'duplicates_filtered': 8.36,
                'predicates_filtered': 8.11,
                'objects': 11.39,
                'pairs': 195.95,
                'max_objects': 119
            },
            'UnRel': {
                'relations': 8.02,
                'duplicates_filtered': 7.89,
                'predicates_filtered': 7.13,
                'objects': 7,
                'pairs': 52.98,
                'max_objects': 21
            },
            'COCO': {
                'relations': 0,
                'duplicates_filtered': 0,
                'predicates_filtered': 0,
                'objects': 12,
                'pairs': 0,
                'max_objects': 110,
            },
        }

    def _set_logger(self):
        """Configure logger."""
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        stream = logging.StreamHandler()
        stream.setFormatter(ColoredFormatter(
            '%(log_color)s%(asctime)s%(reset)s - %(levelname)s - %(message)s'))
        self.logger.addHandler(stream)

    @property
    def annotations_per_batch(self):
        """Return batch size in terms of annotations."""
        if self._batch_size is None or self.task in {'objdet', 'sggen'}:
            return self._annotations_per_batch
        annos_per_img = self._annos_per_img[self.dataset]
        if self.task in {'predcls', 'sgcls'}:
            annos_per_img = annos_per_img['pairs']
        elif self.task == 'objcls':
            annos_per_img = annos_per_img['objects']
        elif self.task == 'preddet' and self.filter_multiple_preds:
            annos_per_img = annos_per_img['predicates_filtered']
        elif self.task == 'preddet' and self.filter_duplicate_rels:
            annos_per_img = annos_per_img['duplicates_filtered']
        elif self.task == 'preddet':
            annos_per_img = annos_per_img['relations']
        return annos_per_img * self._batch_size

    @property
    def batch_size(self):
        """Return batch size in terms of images."""
        if self._batch_size is not None:
            return self._batch_size  # custom batch size defined
        if self.task == 'objdet':
            return 8
        annos_per_img = self._annos_per_img[self.dataset]
        if self.task in {'predcls', 'sgcls'}:
            annos_per_img = annos_per_img['pairs']
        elif self.task == 'objcls':
            annos_per_img = annos_per_img['objects']
        elif self.task == 'preddet' and self.filter_multiple_preds:
            annos_per_img = annos_per_img['predicates_filtered']
        elif self.task == 'preddet' and self.filter_duplicate_rels:
            annos_per_img = annos_per_img['duplicates_filtered']
        elif self.task in {'preddet', 'sggen'}:
            annos_per_img = annos_per_img['relations']
        batch_size = ceil(self._annotations_per_batch / annos_per_img)
        return max(batch_size, 2)

    @property
    def epochs(self):
        """Return number of training epochs."""
        if self._epochs is not None:
            return self._epochs
        return 50 if self.use_early_stopping else 10

    @property
    def logdir(self):
        """Return path of stored Tensorboard logs."""
        return osp.join('runs/', self.net_name, '')

    @property
    def max_obj_dets_per_img(self):
        """Return number of maximum object detections per image."""
        return min(64, self._annos_per_img[self.dataset]['max_objects'])

    @property
    def paths(self):
        """Return a dict of paths useful to train/test/inference."""
        paths = {
            'json_path': self._json_path,
            'models_path': osp.join(self.prerequisites_path,
                                    'models', self.commit, ''),
            'results_path': osp.join(self.prerequisites_path,
                                     'results', self.commit, '')
        }
        for path in paths.values():
            if not osp.exists(path):
                os.mkdir(path)
        return paths

    @property
    def weight_decay(self):
        """Return weight decay for an optimizer."""
        if self._weight_decay is not None:
            return self._weight_decay
        return 5e-5 if 'VG' in self.dataset else 5e-4
