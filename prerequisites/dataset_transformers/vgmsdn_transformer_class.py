# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import json
import os
import shutil
import tarfile

import gdown

from .dataset_transformer_class import DatasetTransformer

VGMSDN = 'https://drive.google.com/uc?id=1RtYidFZRgX1_iYPCaP2buHI1bHacjRTD'


class VGMSDNTransformer(DatasetTransformer):
    """Extands DatasetTransformer for VGMSDN."""

    def __init__(self, config):
        """Initialize VGMSDNTransformer."""
        super().__init__(config)
        self.r1_preds = {
            'carry', 'eat', 'have', 'have a', 'hold', 'wear', 'wear a', 'with'
        }
        self.r2_preds = {
            'at', 'attach to', 'be in', 'be on', 'for', 'hang from', 'hang on',
            'in', 'in a', 'in front of', 'inside', 'inside of', 'lay on',
            'loon at', 'of', 'of a', 'on', 'on a', 'on top of', 'sit in',
            'sit on', 'stand in', 'stand on', 'walk on', 'watch'
        }

    def create_relationship_json(self):
        """Transform VGMSDN annotations."""
        with open(self._orig_annos_path + 'train.json') as fid:
            annos = json.load(fid)
        json_annos = self._merge_rel_annos(annos, 0)
        for anno in json_annos[-2000:]:
            anno['split_id'] = 1
        with open(self._orig_annos_path + 'test.json') as fid:
            annos = json.load(fid)
        json_annos += self._merge_rel_annos(annos, 2)
        noisy_labels = {
            'hang_on': 'hang on',
            'lay_on': 'lay on',
            'hang_from': 'hang from',
            'of_a': 'of a',
            'look_at': 'loon at',
            'in_a': 'in a',
            'walk_on': 'walk on',
            'on_side_of': 'on side of',
            'on_top_of': 'on top of',
            'on_front_of': 'on front of',
            'in_front_of': 'in front of',
            'stand_in': 'stand in',
            'sit_on': 'sit on',
            'inside_of': 'inside of',
            'stand_on': 'stand on',
            'be_in': 'be in',
            'on_a': 'on a',
            'attach_to': 'attach to',
            'next_to': 'next to',
            'wear_a': 'wear a',
            'have_a': 'have a',
            'sit_in': 'sit in',
            'be_on': 'be on'
        }
        for anno in json_annos:
            for rel in anno['relationships']:
                if rel['predicate'] in noisy_labels:
                    rel['predicate'] = noisy_labels[rel['predicate']]
        return json_annos

    def download_annotations(self):
        """Download VGMSDN annotations."""
        if not os.path.exists(self._orig_annos_path):
            os.mkdir(self._orig_annos_path)
        orig_files = ['train.json', 'test.json']
        orig_files = [self._orig_annos_path + name for name in orig_files]
        if not all(os.path.exists(name) for name in orig_files):
            gdown.download(
                VGMSDN,
                self._orig_annos_path + 'top_150_50.tgz', quiet=False
            )
            with tarfile.open(self._orig_annos_path + 'top_150_50.tgz') as fid:
                fid.extractall(self._orig_annos_path)
            os.remove(self._orig_annos_path + 'top_150_50.tgz')
            for name in os.listdir(self._orig_annos_path + 'top_150_50_new/'):
                shutil.move(
                    self._orig_annos_path + 'top_150_50_new/' + name,
                    self._orig_annos_path + name
                )

    def _merge_rel_annos(self, annos, split_id):
        heights_widths = {
            anno['path']: self._compute_im_size(anno['path'])
            for anno in annos
        }
        return [
            {
                'filename': anno['path'],
                'split_id': int(split_id),
                'height': heights_widths[anno['path']][0],
                'width': heights_widths[anno['path']][1],
                'relationships': [
                    {
                        'subject': anno['objects'][rel['sub_id']]['class'],
                        'subject_box': self._decode_box(
                            anno['objects'][rel['sub_id']]['box']),
                        'predicate': str(rel['predicate']),
                        'object': anno['objects'][rel['obj_id']]['class'],
                        'object_box': self._decode_box(
                            anno['objects'][rel['obj_id']]['box']),
                    }
                    for rel in anno['relationships']
                    if self._decode_box(anno['objects'][rel['sub_id']]['box'])
                    and self._decode_box(anno['objects'][rel['obj_id']]['box'])
                ]
            }
            for anno in annos
            if heights_widths[anno['path']][0] is not None
        ]

    @staticmethod
    def _decode_box(box):
        box = [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
        if box[0] >= box[1] or box[2] >= box[3]:
            return []
        return box
