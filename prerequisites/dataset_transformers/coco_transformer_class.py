# -*- coding: utf-8 -*-
"""A class to transform COCO annotations into json format."""

from collections import defaultdict
from copy import deepcopy
import json
import os
import shutil
from zipfile import ZipFile

from tqdm import tqdm

from .dataset_transformer_class import DatasetTransformer

COCO = 'http://images.cocodataset.org/annotations/annotations_trainval2017.zip'


class COCOTransformer(DatasetTransformer):
    """Transform annotations and merge with VRD/VGVTE."""

    def __init__(self, config):
        """Initialize COCOTransformer."""
        super().__init__(config)

    def transform(self):
        """Run the transformation pipeline."""
        jsons = [
            self._predcls_json,
            self._object_json,
            self._word2vec_json
        ]
        if not all(os.path.exists(anno) for anno in jsons):
            self.download_annotations()
            annos = self.create_relationship_json()
            objects = self.save_objects(annos)
            if not os.path.exists(self._word2vec_json):
                self.save_word2vec_vectors(['__background__', 'none'], objects)
            annos = self.update_labels(annos, objects)
            with open(self._predcls_json, 'w') as fid:
                json.dump(annos, fid)

    def create_relationship_json(self):
        """Transform COCO annotations."""
        with open(self._orig_annos_path + 'instances_train2017.json') as fid:
            annos = json.load(fid)
        annos = self._merge_annos(annos, 0)
        with open(self._orig_annos_path + 'instances_val2017.json') as fid:
            val_annos = json.load(fid)
        val_annos = self._merge_annos(val_annos, 2)
        for anno in val_annos[:1000]:
            anno['split_id'] = 1  # keep some for validation
        annos += val_annos
        # Merge with VRD annos (must have transformed VRD first)
        with open(self._preddet_json.replace('COCO', 'VRD')) as fid:
            vrd_annos = json.load(fid)
            for anno in vrd_annos:
                anno['dataset'] = 'VRD'
        annos += vrd_annos
        # Merge with VGVTE annos (must have transformed VGVTE first)
        with open(self._preddet_json.replace('COCO', 'VGVTE')) as fid:
            vg_annos = json.load(fid)
            for anno in vg_annos:
                anno['dataset'] = 'VG'
        annos += vg_annos
        return annos

    def download_annotations(self):
        """Download COCO annotations."""
        if not os.path.exists(self._orig_annos_path):
            os.mkdir(self._orig_annos_path)
        orig_files = ['instances_train2017.json', 'instances_val2017.json']
        orig_files = [self._orig_annos_path + name for name in orig_files]
        if not all(os.path.exists(name) for name in orig_files):
            os.system("wget " + COCO)
            zip_name = self._orig_annos_path + 'annotations_trainval2017.zip'
            shutil.move('annotations_trainval2017.zip', zip_name)
            with ZipFile(zip_name) as fid:
                fid.extractall(self._orig_annos_path)
            os.remove(zip_name)
            shutil.move(
                self._orig_annos_path + 'annotations/instances_train2017.json',
                self._orig_annos_path + 'instances_train2017.json'
            )
            shutil.move(
                self._orig_annos_path + 'annotations/instances_val2017.json',
                self._orig_annos_path + 'instances_val2017.json'
            )
            shutil.rmtree(self._orig_annos_path + 'annotations/')

    def save_objects(self, annos):
        """Save object list."""
        with open('prerequisites/object_categories.json') as fid:
            true_objects = json.load(fid)
        objects = sorted(list(set(
            true_objects[name]
            for anno in annos for name in anno['objects']['names']
            if true_objects[name] is not None
        )))
        with open(self._object_json, 'w') as fid:
            json.dump(objects, fid)
        return objects

    @staticmethod
    def update_labels(annos, objects):
        """Update objects ids."""
        objects = {obj: o for o, obj in enumerate(objects)}
        with open('prerequisites/object_categories.json') as fid:
            true_objects = json.load(fid)
        for anno in annos:
            anno['objects']['boxes'] = [
                box for (box, name) in
                zip(anno['objects']['boxes'], anno['objects']['names'])
                if true_objects[name] is not None
            ]
            anno['objects']['names'] = [
                true_objects[name] for name in anno['objects']['names']
                if true_objects[name] is not None
            ]
            anno['objects']['ids'] = [
                objects[name] for name in anno['objects']['names']
            ]
        return annos

    def _merge_annos(self, annos, split_id):
        images = [
            {
                'id': img['id'],
                'name': img['file_name'],
                'heights_width': self._compute_im_size(img['file_name'])
            }
            for img in tqdm(annos['images'])
        ]
        labels = {label['id']: label['name'] for label in annos['categories']}
        iannos = defaultdict(list)
        for anno in annos['annotations']:
            bbox = list(self._convert_box(anno['bbox']))
            if bbox[1] - bbox[0] >= 1 and bbox[3] - bbox[2] >= 1:
                iannos[anno['image_id']].append(dict({
                    'box': list(bbox),
                    'label': labels[anno['category_id']]
                }))
        return [
            {
                'dataset': 'COCO',
                'filename': img['name'],
                'split_id': split_id,
                'height': img['heights_width'][0],
                'width': img['heights_width'][1],
                'objects': {
                    'boxes': [anno['box'] for anno in iannos[img['id']]],
                    'names': [anno['label'] for anno in iannos[img['id']]]
                },
                'relations': {
                    'names': [], 'ids': [], 'subj_ids': [], 'obj_ids': []
                }
            }
            for img in images if img['heights_width'][0] is not None
        ]

    @staticmethod
    def _convert_box(box):
        """Convert box from [x1, y1, w, h] to [y1, y2, x1, x2]."""
        box = [box[1], box[1] + box[3], box[0], box[0] + box[2]]
        return [int(b) for b in box]
