# -*- coding: utf-8 -*-
"""A class to transform VRD matlab annotations into json format."""

from copy import deepcopy
import os
import shutil
from zipfile import ZipFile

from scipy.io import loadmat

from .dataset_transformer_class import DatasetTransformer

VRD = 'http://cs.stanford.edu/people/ranjaykrishna/vrd/dataset.zip'


class VRDTransformer(DatasetTransformer):
    """Transform matlab annotations to json."""

    def __init__(self, config):
        """Initialize VRDTransformer."""
        super().__init__(config)
        self.r1_preds = {
            'carry', 'contain', 'cover', 'drive', 'eat',
            'feed', 'fly', 'has', 'hit',
            'hold', 'kick', 'play with', 'pull', 'ride',
            'touch', 'use', 'wear', 'with'
        }
        self.r2_preds = {
            'at', 'drive on', 'in', 'inside', 'lean on', 'lying on',
            'on', 'park on',
            'rest on', 'sit on', 'skate on', 'sleep on', 'stand on'
        }

    def create_relationship_json(self):
        """Transform VRD annotations."""
        annos = loadmat(self._orig_annos_path + 'annotation_train.mat')
        json_annos = self._merge_rel_annos(annos['annotation_train'][0], 0)
        annos = loadmat(self._orig_annos_path + 'annotation_test.mat')
        test_annos = self._merge_rel_annos(annos['annotation_test'][0], 2)
        val_annos = deepcopy(test_annos)
        for anno in val_annos:
            anno['split_id'] = 1  # VRD has no validation split
        json_annos += val_annos + test_annos
        return json_annos

    def download_annotations(self):
        """Download VRD annotations."""
        if not os.path.exists(self._orig_annos_path):
            os.mkdir(self._orig_annos_path)
        orig_files = ['annotation_train.mat', 'annotation_test.mat']
        orig_files = [self._orig_annos_path + name for name in orig_files]
        if not all(os.path.exists(name) for name in orig_files):
            os.system("wget " + VRD)
            shutil.move('dataset.zip', self._orig_annos_path + 'dataset.zip')
            with ZipFile(self._orig_annos_path + 'dataset.zip') as fid:
                fid.extractall(self._orig_annos_path)
            os.remove(self._orig_annos_path + 'dataset.zip')
            for name in os.listdir(self._orig_annos_path + 'dataset/'):
                shutil.move(
                    self._orig_annos_path + 'dataset/' + name,
                    self._orig_annos_path + name
                )

    def _merge_rel_annos(self, annos, split_id):
        heights_widths = [
            self._compute_im_size(anno[0]['filename'][0][0])
            for anno in annos
        ]
        return [
            {
                'filename': anno[0]['filename'][0][0],
                'split_id': split_id,
                'height': shw[0],
                'width': shw[1],
                'relationships': [
                    {
                        'subject': rel[0]['phrase'][0][0][0][0],
                        'subject_box': rel[0]['subBox'][0][0].tolist(),
                        'predicate': rel[0]['phrase'][0][0][1][0],
                        'object': rel[0]['phrase'][0][0][2][0],
                        'object_box': rel[0]['objBox'][0][0].tolist()
                    }
                    for rel in anno[0]['relationship'][0][0]
                ]
            }
            for anno, shw in zip(annos, heights_widths)
            if self._handle_no_relationships(anno) and shw[0] is not None
        ]

    @staticmethod
    def _handle_no_relationships(anno):
        """Check if annotation 'anno' has a relationship part."""
        try:
            anno[0]['relationship']
            return True
        except (IndexError, TypeError, ValueError):
            return False
