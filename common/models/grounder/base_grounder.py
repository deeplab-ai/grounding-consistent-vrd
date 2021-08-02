# -*- coding: utf-8 -*-
"""A class to be inherited by other relationship grounders."""

from pdb import set_trace
from copy import deepcopy
import json

import numpy as np
import torch
from torch import nn
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision.ops import MultiScaleRoIAlign

from common.tools import SpatialFeatureExtractor
from common.models.sg_generator import BaseSGGenerator


class BaseGRNDGenerator(BaseSGGenerator):
    """
    Extends PyTorch nn.Module, base class for grouners.

    Inputs:
        - config: Config object, see config.py
        - features: set of str, object features computed on-demand:
            - object_1hots: object 1-hot vectors
            - object_masks: object binary masks
            - pool_features: object pooled features (vectors)
            - roi_features: object pre-pooled features (volumes)
    """

    def __init__(self, config, features, **kwargs):
        """Initialize layers."""
        super().__init__(config, features, **kwargs)

    def forward(self, image, object_boxes, predicate_ids, object_ids,
                pairs, image_info):
        """
        Forward pass.

        Expects:
            - image: image tensor, (3 x H x W) or None
            - object_boxes: tensor, (n_obj, 4), (xmin, ymin, xmax, ymax)
            - predicate_ids: tensor, (n_rel), predicate category of pairs
            - object_ids: tensor, (n_obj,), object category ids
            - pairs: tensor, (n_rel, 2), pairs of objects to examine
            - image_info: tuple, (im_width, im_height)
        """
        # Base features
        self._image_info = image_info
        base_features = None
        if 'base_features' in self.features:
            base_features = self.get_base_features(image)
        # Object features
        objects = {'boxes': object_boxes, 'ids': object_ids,
                   'norm_boxes': self.get_norm_boxes(object_boxes, image_info)}
        if 'pool_features' in self.features:
            objects['pool_features'] = self.get_obj_pooled_features(
                base_features, object_boxes
            )
        if 'roi_features' in self.features:
            objects['roi_features'] = self.get_roi_features(
                base_features, object_boxes
            )
        if 'object_1hots' in self.features:
            objects['1hots'] = self.get_obj_1hot_vectors(object_ids)
        if 'object_masks' in self.features:
            objects['masks'] = self.get_obj_masks(object_boxes)
        # Iterative forward pass over sub-batches for memory issues
        outputs = [
            self.net_forward(
                base_features, objects,
                pairs[range(
                    btch * self.rel_batch_size,
                    min((btch + 1) * self.rel_batch_size, len(pairs))
                )],
                predicate_ids[range(
                    btch * self.rel_batch_size,
                    min((btch + 1) * self.rel_batch_size, len(pairs))
                )]
            )
            for btch in range(1 + (len(pairs) - 1) // self.rel_batch_size)
        ]
        return [
            torch.cat([output[k] for output in outputs], dim=0)
            if outputs[0][k] is not None else None
            for k in range(len(outputs[0]))
        ]

    def get_norm_boxes(self, boxes, image_info):
        """
        Normalize boxes according to Collell(2018).

            - boxes: tensor, (n_obj, 4), (xmin, ymin, xmax, ymax)
            - image_info: (Width, Height)
        """
        W, H = image_info
        boxes_norm = torch.zeros_like(boxes)
        boxes_norm[:, 2] = 0.5 * (boxes[:, 2] - boxes[:, 0]) / W
        boxes_norm[:, 3] = 0.5 * (boxes[:, 3] - boxes[:, 1]) / H
        boxes_norm[:, 0] = boxes[:, 0] / W + boxes_norm[:, 2]
        boxes_norm[:, 1] = boxes[:, 1] / H + boxes_norm[:, 3]
        return boxes_norm
