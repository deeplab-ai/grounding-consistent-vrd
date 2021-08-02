# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import json
import math
import os
import shutil

import numpy as np
import h5py

from .dataset_transformer_class import DatasetTransformer


class VG200Transformer(DatasetTransformer):
    """Extends DatasetTransformer for VG200 annotations."""

    def __init__(self, config):
        """Initialize VG200Transformer."""
        super().__init__(config)
        self.r1_preds = {
            'carrying', 'eating', 'has', 'holding', 'playing',
            'riding', 'using', 'wearing', 'wears', 'with'
        }
        self.r2_preds = {
            'at', 'attached to', 'belonging to', 'flying in',
            'for', 'from', 'growing on', 'hanging from', 'in',
            'laying on', 'looking at', 'lying on', 'made of',
            'mounted on', 'of', 'on', 'painted on', 'parked on',
            'part of', 'says', 'sitting on', 'standing on', 'to',
            'walking in', 'walking on', 'watching'
        }

    def create_relationship_json(self):
        """
        Transform VG200 annotations.

        Inputs:
            - [
                {
                    'filename': name,
                    'split_id': split_id,
                    'relationships': {rel_id: pair},
                    'boxes': {obj_id: decoded box}
                }
            ]
        """
        self._load_dataset()
        return self._set_annos()

    def download_annotations(self):
        """Download VG200 annotations."""
        if not os.path.exists(self._orig_annos_path):
            os.mkdir(self._orig_annos_path)
        orig_files = ['image_data.json', 'VG-SGG-dicts.json', 'VG-SGG.h5']
        orig_files = [self._orig_annos_path + name for name in orig_files]
        if not all(os.path.exists(name) for name in orig_files):
            os.system("wget http://svl.stanford.edu/projects/scene-graph/VG/image_data.json")
            os.system("wget http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG-dicts.json")
            os.system("wget http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5")
            shutil.move(
                'image_data.json',
                self._orig_annos_path + 'image_data.json'
            )
            shutil.move(
                'VG-SGG-dicts.json',
                self._orig_annos_path + 'VG-SGG-dicts.json'
            )
            shutil.move('VG-SGG.h5', self._orig_annos_path + 'VG-SGG.h5')

    def _load_dataset(self):
        # Load images' metadata
        with open(self._orig_annos_path + 'image_data.json') as fid:
            self._img_names = [
                img['url'].split('/')[-1]
                for img in json.load(fid)
                if img['image_id'] not in [1592, 1722, 4616, 4617]
            ]

        # Load object and predicate names
        with open(self._orig_annos_path + 'VG-SGG-dicts.json') as fid:
            dict_annos = json.load(fid)
        self._predicate_names = {
            int(key): val
            for key, val in dict_annos['idx_to_predicate'].items()
        }
        self._object_names = {
            int(key): val
            for key, val in dict_annos['idx_to_label'].items()
        }

    @staticmethod
    def _transform_annotations(annos):
        """Return input, dummy implementation for consistency."""
        return annos

    def _set_annos(self):
        annos = h5py.File(self._orig_annos_path + 'VG-SGG.h5', 'r')
        split_ids = np.array(annos['split'])
        first_test_index = np.nonzero(split_ids)[0][0]
        split_ids[first_test_index - 2000: first_test_index] = 1
        boxes = np.array(annos['boxes_512'])
        obj_labels = [
            int(label) for label in np.array(annos['labels']).flatten()]
        predicate_labels = [
            int(pred) for pred in np.array(annos['predicates']).flatten()]
        relationships = np.array(annos['relationships'])
        heights_widths = {
            img_name: self._compute_im_size(img_name)
            for img_name in self._img_names
        }
        annos = [
            {
                'filename': name,
                'split_id': int(split_id),
                'height': heights_widths[name][0],
                'width': heights_widths[name][1],
                'objects': {
                    'names': [
                        self._object_names[obj_labels[obj]]
                        for obj in range(first_box, last_box + 1)],
                    'boxes': [
                        self._decode_box(
                            boxes[obj],
                            heights_widths[name][0],
                            heights_widths[name][1],
                            512)
                        for obj in range(first_box, last_box + 1)]
                },
                'relations': {
                    'names': [
                        self._predicate_names[predicate_labels[rel]]
                        for rel in range(first_rel, last_rel + 1)
                        if first_rel > -1],
                    'subj_ids': [
                        int(relationships[rel][0] - first_box)
                        for rel in range(first_rel, last_rel + 1)
                        if first_rel > -1],
                    'obj_ids': [
                        int(relationships[rel][1] - first_box)
                        for rel in range(first_rel, last_rel + 1)
                        if first_rel > -1]
                }
            }
            for name, split_id, first_rel, last_rel, first_box, last_box
            in zip(
                self._img_names, split_ids, annos['img_to_first_rel'][:],
                annos['img_to_last_rel'][:], annos['img_to_first_box'][:],
                annos['img_to_last_box'][:]
            )
            if first_box > -1 and heights_widths[name][0] is not None
        ]
        return annos

    @staticmethod
    def _decode_box(box, orig_height, orig_width, im_long_size):
        """
        Convert encoded box back to original.

        Inputs:
            - box: array, [x_center, y_center, width, height]
            - orig_height: int, height of the original image
            - orig_width: int, width of the original image
            - im_long_size: int, rescaled length of longer lateral
        Returns:
            - decoded box: list, [y_min, y_max, x_min, x_max]
        """
        # Center-oriented to left-top-oriented
        box = box.tolist()
        box[0] -= box[2] / 2
        box[1] -= box[3] / 2

        # Re-scaling to original size
        scale = max(orig_height, orig_width) / im_long_size
        box[0] = max(math.floor(scale * box[0]), 0)
        box[1] = max(math.floor(scale * box[1]), 0)
        box[2] = max(math.ceil(scale * box[2]), 2)
        box[3] = max(math.ceil(scale * box[3]), 2)

        # Boxes at least 2x2 that fit in the image
        box[0] = min(box[0], orig_width - 2)
        box[1] = min(box[1], orig_height - 2)
        box[2] = min(box[2], orig_width - box[0])
        box[3] = min(box[3], orig_height - box[1])

        # Convert to [y_min, y_max, x_min, x_max]
        return [box[1], box[1] + box[3] - 1, box[0], box[0] + box[2] - 1]
