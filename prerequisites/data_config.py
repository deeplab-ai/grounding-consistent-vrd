# -*- coding: utf-8 -*-
"""Configuration parameters for each dataset."""

import os
from os import path as osp


class DataConfig:
    """A class to configure global parameters."""

    def __init__(self, data_path, dataset='VRD'):
        """Initialize configuration instance."""
        self._data_path = data_path
        self.dataset = dataset

    @property
    def glove_txt(self):
        """Return a dict of paths useful to train/test/inference."""
        return osp.join(self._data_path, 'glove.42B.300d.txt')

    @property
    def orig_annos_path(self):
        """Return path of stored original dataset annotations."""
        if not osp.exists('prerequisites/datasets/'):
            os.mkdir('prerequisites/datasets/')
        return osp.join('prerequisites/datasets/', self.dataset, '')

    @property
    def orig_img_path(self):
        """Return path of stored dataset images."""
        _dataset = 'VG' if 'VG' in self.dataset else self.dataset
        return osp.join(self._data_path, _dataset, 'images', '')

    @property
    def paths(self):
        """Return a dict of paths useful to train/test/inference."""
        paths = {'json_path': osp.join(self._data_path, 'sgg_annos/')}
        for path in paths.values():
            if not osp.exists(path):
                os.mkdir(path)
        return paths
