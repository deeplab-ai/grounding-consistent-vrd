# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import os

import gdown
import h5py

from .dataset_transformer_class import DatasetTransformer

VGVTE = 'https://drive.google.com/uc?id=1C6MDiqWQupMrPOgk4T12zWiAJAZmY1aa'


class VGVTETransformer(DatasetTransformer):
    """Extands DatasetTransformer for VGVTE."""

    def __init__(self, config):
        """Initialize VGVTETransformer."""
        super().__init__(config)
        self.r1_preds = {
            'carry', 'catch', 'contain', 'cut', 'eat', 'fly', 'have',
            'hit', 'hold', 'play', 'pull', 'ride', 'show', 'throw',
            'use', 'wear', 'with'
        }
        self.r2_preds = {
            'adorn', 'against', 'at', 'attach to', 'belong to', 'build into',
            'drive on', 'fill with', 'fly in', 'for', 'from', 'grow in',
            'grow on', 'hang in', 'hang on', 'hold by', 'in', 'inside',
            'lay in', 'lay on', 'lean on', 'look at', 'mount on', 'of', 'on',
            'on top of', 'paint on', 'park', 'part of', 'print on',
            'reflect in', 'rest on', 'say', 'sit at', 'sit in', 'sit on',
            'stand on', 'standing in', 'to', 'walk', 'walk in', 'walk on',
            'watch', 'wear by', 'write on'
        }

    def create_relationship_json(self):
        """Transform VGVTE annotations."""
        annos = h5py.File(self._orig_annos_path + 'vg1_2_meta.h5', 'r')
        self._predicates = {
            int(idx): str(name[()])
            for idx, name in dict(annos['meta']['pre']['idx2name']).items()
        }
        self._objects = {
            int(idx): str(name[()])
            for idx, name in dict(annos['meta']['cls']['idx2name']).items()
        }
        json_annos = self._merge_rel_annos(dict(annos['gt']['train']), 0)
        for anno in json_annos[-2000:]:
            anno['split_id'] = 1
        json_annos += self._merge_rel_annos(dict(annos['gt']['test']), 2)
        noisy_images = {  # contain bboxes > image
            '1829.jpg', '2391277.jpg', '150333.jpg',
            '3201.jpg', '713208.jpg', '1592325.jpg'
        }
        json_annos = [
            anno for anno in json_annos if anno['filename'] not in noisy_images
        ]
        return json_annos

    def download_annotations(self):
        """Download VGVTE annotations."""
        if not os.path.exists(self._orig_annos_path):
            os.mkdir(self._orig_annos_path)
        if not os.path.exists(self._orig_annos_path + 'vg1_2_meta.h5'):
            gdown.download(
                VGVTE,
                self._orig_annos_path + 'vg1_2_meta.h5', quiet=False
            )

    def _merge_rel_annos(self, annos, split_id):
        heights_widths = {
            filename: self._compute_im_size(filename + '.jpg')
            for filename in annos.keys()
        }
        return [
            {
                'filename': filename + '.jpg',
                'split_id': int(split_id),
                'height': heights_widths[filename][0],
                'width': heights_widths[filename][1],
                'relationships': [
                    {
                        'subject': self._objects[rel[0]].lower(),
                        'subject_box': self._decode_box(sub_box),
                        'predicate': self._predicates[rel[1]].lower(),
                        'object': self._objects[rel[2]].lower(),
                        'object_box': self._decode_box(obj_box)
                    }
                    for sub_box, rel, obj_box in zip(
                        relationships['sub_boxes'],
                        relationships['rlp_labels'],
                        relationships['obj_boxes'])
                ]
            }
            for filename, relationships in annos.items()
            if heights_widths[filename][0] is not None
        ]

    @staticmethod
    def _decode_box(box):
        return [int(box[1]), int(box[3]), int(box[0]), int(box[2])]
