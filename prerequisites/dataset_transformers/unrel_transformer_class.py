# -*- coding: utf-8 -*-
"""A class to transform UnRel matlab annotations into json format."""

import os
import shutil
import tarfile

from scipy.io import loadmat

from .dataset_transformer_class import DatasetTransformer

UNREL = 'http://www.di.ens.fr/willow/research/unrel/data/unrel-dataset.tar.gz'


class UnRelTransformer(DatasetTransformer):
    """Transform matlab annotations to json."""

    def __init__(self, config):
        """Initialize VRDTransformer."""
        super().__init__(config)

    def create_relationship_json(self):
        """Transform VRD annotations."""
        annos = loadmat(self._orig_annos_path + 'annotations.mat')
        json_annos = self._merge_unrel_annos(annos['annotations'])
        vrd_annos_path = self._orig_annos_path.replace('UnRel', 'VRD')
        annos = loadmat(vrd_annos_path + 'annotation_train.mat')
        json_annos += self._merge_vrd_annos(annos['annotation_train'][0], 0)
        annos = loadmat(vrd_annos_path + 'annotation_test.mat')
        json_annos += self._merge_vrd_annos(annos['annotation_test'][0], 1)
        return json_annos

    def download_annotations(self):
        """Download UnRel annotations."""
        if not os.path.exists(self._orig_annos_path):
            os.mkdir(self._orig_annos_path)
        if not os.path.exists(self._orig_annos_path + 'annotations.mat'):
            os.system("wget " + UNREL)
            stored_tar = self._orig_annos_path + 'unrel-dataset.tar.gz'
            shutil.move('unrel-dataset.tar.gz', stored_tar)
            with tarfile.open(stored_tar) as fid:
                fid.extractall(self._orig_annos_path)
            os.remove(stored_tar)
            shutil.rmtree(self._orig_annos_path + 'images')

    def _merge_unrel_annos(self, annos):
        heights_widths = [
            self._compute_im_size(anno[0][0][0][0][0])
            for anno in annos
        ]
        return [  # ignore corrupted image 1124.jpg
            {
                'filename': anno[0][0][0][0][0],
                'split_id': 2,
                'height': shw[0],
                'width': shw[1],
                'relationships': [
                    {
                        'subject': rel[0][0][0][0][0],
                        'subject_box': self._decode_box(rel[0][0][0][2][0]),
                        'predicate': rel[0][0][0][4][0][0][0],
                        'object': rel[0][0][0][1][0],
                        'object_box': self._decode_box(rel[0][0][0][3][0])
                    }
                    for rel in anno[0][0][0][2]
                    if len(rel[0][0][0][3][0]) == 4
                    and len(rel[0][0][0][2][0]) == 4
                ]
            }
            for anno, shw in zip(annos, heights_widths)
            if shw[0] is not None and anno[0][0][0][0][0] != '1124.jpg'
        ]

    @staticmethod
    def _decode_box(box, rel=None):
        box = box.tolist()
        return [int(box[1]), int(box[3]), int(box[0]), int(box[2])]

    def _merge_vrd_annos(self, annos, split_id):
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
