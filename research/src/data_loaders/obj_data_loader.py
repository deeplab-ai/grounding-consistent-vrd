# -*- coding: utf-8 -*-
"""Custom dataset for Object Detection/Classification."""

from .base_data_loader import BaseDataset


class ObjDataset(BaseDataset):
    """Dataset utilities for Object Detection/Classification."""

    def __init__(self, annotations, config, features={}):
        """Initialize dataset."""
        super().__init__(annotations, config, features)
        self._features = features.union({
            'image_info', 'images', 'object_ids', 'object_rois'
        })
