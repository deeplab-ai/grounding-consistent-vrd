# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import json
import os
import shutil
from zipfile import ZipFile

from .dataset_transformer_class import DatasetTransformer


class VG80KTransformer(DatasetTransformer):
    """Extands DatasetTransformer for VG80K."""

    def __init__(self, config):
        """Initialize VG80kTransformer."""
        super().__init__(config)

    def create_relationship_json(self):
        """Transform VG80K annotations."""
        with open(self._orig_annos_path + 'train_clean.json') as fid:
            split_ids = {img: 0 for img in json.load(fid)}
        with open(self._orig_annos_path + 'val_clean.json') as fid:
            split_ids.update({img: 1 for img in json.load(fid)})
        with open(self._orig_annos_path + 'test_clean.json') as fid:
            split_ids.update({img: 2 for img in json.load(fid)})
        anno_json = 'relationships_clean_spo_joined_and_merged.json'
        with open(self._orig_annos_path + anno_json) as fid:
            annos = json.load(fid)
        self.noisy_labels = {
            'sasani', 'skyramp', 'linespeople', 'buruburu',
            'gunport', 'dirttrack', 'greencap', 'shrublike'
        }
        json_annos = self._merge_rel_annos(annos, split_ids)
        return json_annos

    def download_annotations(self):
        """Download VG80K annotations."""
        if not os.path.exists(self._orig_annos_path):
            os.mkdir(self._orig_annos_path)
            # Download, unzip and remove zip file
            os.system(
                "wget https://www.dropbox.com/s/minpyv59crdifk9/datasets.zip"
            )
            shutil.move('datasets.zip', self._orig_annos_path + 'datasets.zip')
            with ZipFile(self._orig_annos_path + 'datasets.zip') as fid:
                fid.extractall(self._orig_annos_path)
            os.remove(self._orig_annos_path + 'datasets.zip')
            # Move json annotations to given path
            folder = 'datasets/large_scale_VRD/Visual_Genome/'
            for name in os.listdir(self._orig_annos_path + folder):
                if name.endswith('.json'):
                    shutil.move(
                        self._orig_annos_path + folder + name,
                        self._orig_annos_path + name
                    )
            shutil.rmtree(self._orig_annos_path + 'datasets')

    def _merge_rel_annos(self, annos, split_ids):
        heights_widths = {
            anno['image_id']:
                self._compute_im_size(str(anno['image_id']) + '.jpg')
            for anno in annos
        }
        return [
            {
                'filename': str(anno['image_id']) + '.jpg',
                'split_id': int(split_ids[anno['image_id']]),
                'height': heights_widths[anno['image_id']][0],
                'width': heights_widths[anno['image_id']][1],
                'relationships': [
                    {
                        'subject': rel['subject']['name'],
                        'subject_box': self._decode_box([
                            rel['subject'][item]
                            for item in ['x', 'y', 'w', 'h']]),
                        'predicate': str(rel['predicate']),
                        'object': rel['object']['name'],
                        'object_box': self._decode_box([
                            rel['object'][item]
                            for item in ['x', 'y', 'w', 'h']])
                    }
                    for rel in anno['relationships']
                    if self._decode_box([
                        rel['subject'][item] for item in ['x', 'y', 'w', 'h']])
                    and self._decode_box([
                        rel['object'][item] for item in ['x', 'y', 'w', 'h']])
                    and all(
                        word not in self.noisy_labels
                        for word in rel['subject']['name'].split())
                    and all(
                        word not in self.noisy_labels
                        for word in rel['object']['name'].split())
                    and all(
                        word not in self.noisy_labels
                        for word in rel['predicate'].split())
                ]
            }
            for anno in annos
            if heights_widths[anno['image_id']][0] is not None
        ]

    @staticmethod
    def _decode_box(box):
        box = [
            int(box[1]), int(box[1]) + int(box[3]),
            int(box[0]), int(box[0]) + int(box[2])
        ]
        if box[0] >= box[1] or box[2] >= box[3]:
            return []
        return box
