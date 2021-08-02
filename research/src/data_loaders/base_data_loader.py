# -*- coding: utf-8 -*-
"""Custom datasets and data loaders for Scene Graph Generation."""

from copy import deepcopy
import json
import random

import numpy as np
import PIL
import torch
from torch import multiprocessing
from torch.utils.data import Dataset
import torchvision
# torch.multiprocessing.set_sharing_strategy('file_system')


class BaseDataset(Dataset):
    """Dataset utilities for Scene Graph Generation."""

    def __init__(self, annotations, config, features={}):
        """
        Initialize dataset.

        Inputs:
            - annotations: list of annotations per image
            - config: config class, see config.py
            - features: set of str, features to use in train/test
        """
        self._annotations = annotations
        self._config = config
        self._features = features
        self._set_init()
        self._set_methods()

    def __getitem__(self, index):
        """Get image's data (used by loader to later form a batch)."""
        anno = deepcopy(self._annotations[index])
        if not self._task.startswith('obj'):
            anno = self._add_background(anno)  # add/filter bg pairs
        # Augment dataset by distorting images/boxes
        self.jitter = 0
        self.grayscale = 0
        self.flip = 0
        if self._augment_annotations and anno['split_id'] == 0:
            self.jitter = 0.25 if random.random() < 0.5 else 0
            self.grayscale = 0.15 if random.random() < 0.5 else 0
            self.flip = 1 if random.random() < 0.5 else 0
            if self.flip == 1:  # reverse image annotations
                anno = self._reverse_anno(anno)
            if random.random() < 0.5:  # distort boxes
                anno = self._distort_boxes(anno)
        # Create json of features for this sample
        return_json = {
            feature: self._methods[feature](anno)
            for feature in self._features if feature in self._methods
        }
        return_json['filenames'] = anno['filename']
        return return_json

    def __len__(self):
        """Override __len__ method, return dataset's size."""
        return len(self._annotations)

    def _set_init(self):
        """Set dataset variables."""
        self._augment_annotations = self._config.augment_annotations
        self._bg_perc = self._config.bg_perc
        self._dataset = self._config.dataset
        self._json_path = self._config.paths['json_path']
        self._orig_image_path = self._config.orig_img_path
        self._predicates = np.array(self._config.rel_classes)
        self._task = self._config.task
        self._is_set = {
            'plausibilities': False,
            'probabilities': False,
            'similarities': False
        }
        self._config.logger.debug(
            'Set up dataset of %d files' % len(self._annotations))

    def _set_methods(self):
        """Correspond a method to each feature type."""
        self._methods = {
            'bg_targets': self.get_bg_targets,
            'boxes': self.get_boxes,
            'images': self.get_image,
            'image_info': self.get_image_info,
            'labels': self.get_labels,
            'object_scores': self.get_object_scores,
            'negative_ids': self.get_negative_ids,
            'object_ids': self.get_object_ids,
            'object_rois': self.get_object_rois,
            'object_rois_norm': self.get_norm_object_rois,
            'pairs': self.get_pairs,
            'predicate_ids': self.get_predicate_ids,
            'predicate_plausibilities': self.get_predicate_plausibilities,
            'predicate_probabilities': self.get_predicate_probabilities,
            'predicate_similarities': self.get_predicate_similarities
        }

    def _set_plausibilities(self, mode='predcls'):
        """Set predicate plausibility matrix for given dataset."""
        json_name = ''.join([
            self._json_path, self._dataset, '_plausibilities.json'])
        with open(json_name) as fid:
            plausibilities = np.array(json.load(fid))
        inds = np.argsort(plausibilities, 2)[:, :, ::-1]
        (objs, _, preds) = plausibilities.shape
        neg_inds = inds[:, :, int(preds / 2):].reshape(objs ** 2, -1)
        pos_inds = inds[:, :, :int(preds / 2)].reshape(objs ** 2, -1)
        plausibilities = plausibilities.reshape(-1, preds)
        plausibilities[np.arange(len(plausibilities))[:, None], neg_inds] = 0
        plausibilities[np.arange(len(plausibilities))[:, None], pos_inds] = -1
        self.plausibilities = plausibilities.reshape(objs, objs, -1)
        self._is_set['plausibilities'] = True

    def _set_probabilities(self, mode='predcls'):
        """Set predicate probability matrix for given dataset."""
        json_name = ''.join([
            self._json_path, self._dataset, '_', mode, '_probabilities.json'])
        with open(json_name) as fid:
            self.probabilities = np.array(json.load(fid))
        self._is_set['probabilities'] = True

    def _set_similarities(self):
        """Set predicate probability matrix for given dataset."""
        json_name = self._json_path + self._dataset + '_pred_cooccurences.json'
        with open(json_name) as fid:
            pred_similarities = json.load(fid)
        if self._dataset != 'VrR-VG':
            json_name = (
                self._json_path + self._dataset + '_spatial_similarities.json'
            )
            with open(json_name) as fid:
                spatial_similarities = json.load(fid)
        else:
            spatial_similarities = pred_similarities
        self.similarities = {
            key: (
                np.array(spatial_similarities[key])
                + np.array(pred_similarities[key])
            ) * 0.5
            for key in spatial_similarities
        }
        self._is_set['similarities'] = True

    @staticmethod
    def get_bg_targets(anno):
        """Return foreground/background targets for given image."""
        return np.array([
            0 if pred == '__background__' else 1
            for pred in anno['relations']['names']])

    @staticmethod
    def get_boxes(anno):
        """Return (N, 2, 4) bounding boxes for given image."""
        boxes = anno['objects']['boxes']
        return np.concatenate((
            boxes[anno['relations']['subj_ids']][:, None, :],
            boxes[anno['relations']['obj_ids']][:, None, :]), axis=1)

    def get_image(self, anno):
        """Return an image blob (1, H, W, 3)."""
        preprocessing = torchvision.transforms.Compose([
            torchvision.transforms.ColorJitter(
                self.jitter, self.jitter, self.jitter, self.jitter),
            torchvision.transforms.RandomGrayscale(self.grayscale),
            torchvision.transforms.RandomHorizontalFlip(self.flip),
            torchvision.transforms.ToTensor()
        ])
        img_folder = self._orig_image_path
        if 'dataset' in anno and anno['dataset'] is not None:
            img_folder = img_folder.replace(self._dataset, anno['dataset'])
        return preprocessing(PIL.Image.open(img_folder + anno['filename']))

    @staticmethod
    def get_image_info(anno):
        """Return a whole given image as a RoI."""
        return (anno['width'], anno['height'])

    @staticmethod
    def get_labels(anno):
        """Return label vector (subj, -1, obj) for given image."""
        object_ids = anno['objects']['ids']
        labels = -np.ones((len(anno['relations']['subj_ids']), 3))
        labels[:, 0] = object_ids[anno['relations']['subj_ids']]
        labels[:, 2] = object_ids[anno['relations']['obj_ids']]
        return labels

    @staticmethod
    def get_object_scores(anno):
        """Return (if present) the object detector scores (for sggen)"""
        return anno['objects']['scores']

    @staticmethod
    def get_negative_ids(anno):
        """Return negative predicate ids for given image."""
        return np.array([
            -1 if not neg_ids else neg_ids[0]
            for neg_ids in anno['relations']['neg_ids']
        ])

    @staticmethod
    def get_object_ids(anno):
        """Return object ids for given image."""
        return anno['objects']['ids']

    @staticmethod
    def get_object_rois(anno):
        """Return rois for objects of given image (for SGCls)."""
        boxes = anno['objects']['boxes']
        return np.round(boxes[:, (2, 0, 3, 1)])

    @staticmethod
    def get_norm_object_rois(anno):
        """Return normalized object rois according to Collell (2018)."""
        boxes = anno['objects']['boxes'].astype(float)
        boxes = boxes[:, (2, 0, 3, 1)]
        boxes_norm = np.zeros_like(boxes)
        boxes_norm[:, 2] = 0.5 * (boxes[:, 2] - boxes[:, 0]) / anno['width']
        boxes_norm[:, 3] = 0.5 * (boxes[:, 3] - boxes[:, 1]) / anno['height']
        boxes_norm[:, 0] = 0.5 * (boxes[:, 2] + boxes[:, 0]) / anno['width']
        boxes_norm[:, 1] = 0.5 * (boxes[:, 3] + boxes[:, 1]) / anno['height']
        return boxes_norm

    @staticmethod
    def get_pairs(anno):
        """Return an array of related object ids for given image."""
        return np.stack(
            (anno['relations']['subj_ids'], anno['relations']['obj_ids']),
            axis=1
        )

    @staticmethod
    def get_predicate_ids(anno):
        """Return predicate ids for given image."""
        return anno['relations']['ids']

    def get_predicate_plausibilities(self, anno):
        """Return predicate plausibility vectors for given image."""
        if not self._is_set['plausibilities']:
            self._set_plausibilities()
        object_ids = anno['objects']['ids']
        return self.plausibilities[
            object_ids[anno['relations']['subj_ids']],
            object_ids[anno['relations']['obj_ids']]
        ]

    def get_predicate_probabilities(self, anno, mode='predcls'):
        """Return predicate probability vectors for given image."""
        if not self._is_set['probabilities']:
            self._set_probabilities(mode)
        object_ids = anno['objects']['ids']
        return self.probabilities[
            object_ids[anno['relations']['subj_ids']],
            object_ids[anno['relations']['obj_ids']]
        ]

    def get_predicate_similarities(self, anno):
        """Return predicate similarity vectors for given image."""
        if not self._is_set['similarities']:
            self._set_similarities()
        object_names = anno['objects']['names']
        sims = np.stack([
            self.similarities['_'.join([subj, pred, obj])]
            if '_'.join([subj, pred, obj]) in self.similarities
            else (
                np.array([0] * (len(self._config.rel_classes) - 1) + [1])
                if pred == '__background__'
                else -np.ones(len(self._config.rel_classes))
            )
            for subj, pred, obj in zip(
                object_names[anno['relations']['subj_ids']],
                anno['relations']['names'],  # np.maximum(anno['relations']['ids'], anno['relations']['neg_ids']),
                object_names[anno['relations']['obj_ids']]
            )])
        pls = self.get_predicate_plausibilities(anno)
        return np.maximum(sims, pls)

    def _add_background(self, anno):
        """Add some background annotations."""
        anno = deepcopy(anno)
        inds = np.array([
            n for n, name in enumerate(anno['relations']['names'])
            if name != '__background__'
            or anno['relations']['neg_ids'][n]
            or (anno['split_id'] == 0 and random.random() < self._bg_perc)
            or (anno['split_id'] != 0 and self._task != 'preddet')
        ])
        if not inds.tolist():
            inds = np.array([0])
        anno['relations']['names'] = anno['relations']['names'][inds]
        anno['relations']['ids'] = anno['relations']['ids'][inds]
        anno['relations']['subj_ids'] = anno['relations']['subj_ids'][inds]
        anno['relations']['obj_ids'] = anno['relations']['obj_ids'][inds]
        anno['relations']['neg_ids'] = anno['relations']['neg_ids'][inds]
        return anno

    @staticmethod
    def _distort_boxes(anno):
        """Distort boxes to augment dataset."""
        anno = deepcopy(anno)
        boxes = anno['objects']['boxes']
        multipliers = (2 * np.random.random(boxes.shape) - 1) * 0.07 + 1
        new_boxes = np.round(multipliers * boxes)
        new_boxes[:2][new_boxes[:2] > anno['height'] - 1] = anno['height'] - 1
        new_boxes[2:][new_boxes[2:] > anno['width'] - 1] = anno['width'] - 1
        new_boxes[:, 0] = np.minimum(new_boxes[:, 0], new_boxes[:, 1] - 2)
        new_boxes[:, 2] = np.minimum(new_boxes[:, 2], new_boxes[:, 3] - 2)
        anno['objects']['boxes'] = new_boxes
        return anno

    def _reverse_anno(self, anno):
        """Reverse anno supposing a vertical flip."""
        anno = deepcopy(anno)
        boxes = np.copy(anno['objects']['boxes'])
        boxes[:, 2:] = anno['width'] - boxes[:, (3, 2)]
        anno['objects']['boxes'] = boxes
        if self._dataset == 'VRD':
            anno['relations']['ids'] = np.array([
                (34 if pred == 35 else 35) if pred in {34, 35} else pred
                for pred in anno['relations']['ids']
            ])
        return anno


def sgg_collate_fn(batch_data):
    """Collate function for custom data loading."""
    return_batch = {}
    tensor_features = {
        'object_rois',
        'object_rois_norm',
        'predicate_probabilities',
        'predicate_similarities',
        'object_scores'
    }
    for feature in batch_data[0].keys():
        if feature in tensor_features:
            return_batch[feature] = [
                torch.from_numpy(item[feature]).float() for item in batch_data]
        elif 'targets' in feature or 'ids' in feature:  # ids are long integers
            return_batch[feature] = [
                torch.from_numpy(item[feature]).long() for item in batch_data]
        else:  # list of numpy arrays
            return_batch[feature] = [item[feature] for item in batch_data]
    return return_batch


class SGGDataLoader(torch.utils.data.DataLoader):
    """Custom data loader for Scene Graph Generation."""

    def __init__(self, dataset, batch_size, shuffle=True, num_workers=2,
                 drop_last=False, device='cuda:0'):
        """Initialize loader for given dataset and annotations."""
        super().__init__(
            dataset, batch_size=batch_size, shuffle=shuffle,
            num_workers=num_workers, drop_last=drop_last,
            collate_fn=lambda data: sgg_collate_fn(data))
        self._device = device

    def get(self, feature, batch, step):
        """Get specific feature from a given batch."""
        not_tensors = {'boxes', 'filenames', 'labels', 'pairs', 'image_info'}
        if feature not in batch:
            return None
        if feature in not_tensors:
            return batch[feature][step]
        return batch[feature][step].to(self._device)
