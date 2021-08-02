# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import json
import os
import random
import shutil
import xml.etree.ElementTree as ET
from zipfile import ZipFile

import gdown
import h5py

from .dataset_transformer_class import DatasetTransformer


class VrRVGTransformer(DatasetTransformer):
    """Extands DatasetTransformer for VrR-VG."""

    def __init__(self, config):
        """Initialize VrRVGTransformer."""
        super().__init__(config)

    def create_relationship_json(self):
        """Transform VrR-VG annotations."""
        annos = [
            self._xml_to_dict(
                ET.parse(self._orig_annos_path + filename).getroot())
            for filename in os.listdir(self._orig_annos_path)
            if filename.endswith('.xml')
        ]
        self._set_split_ids()
        heights_widths = {
            anno['filename']: self._compute_im_size(anno['filename'])
            for anno in annos
        }
        objects = [
            {obj['object_id']: obj for obj in anno['objects']}
            for anno in annos
        ]
        corrupted_images = {  # corrupted images or bboxes > image
            '1592', '1722', '4616', '4617', '498042', '150333', '3201'
        }
        self._noisy_names = {
            "t shirt": "t_shirt",
            "t-shirt": "t_shirt",
            "tshirt": "t_shirt",
            "tee shirt": "t_shirt",
            "doughnut": "donut",
            "doughnuts": "donuts",
            "grey": "gray"
        }
        return [
            {
                'filename': anno['filename'],
                'split_id': int(self._split_ids[anno['filename']]),
                'height': heights_widths[anno['filename']][0],
                'width': heights_widths[anno['filename']][1],
                'relationships': [
                    {
                        'subject': self._denoise_names(
                            object_dict[rel['subject_id']]['name']),
                        'subject_box': self._decode_box(
                            object_dict[rel['subject_id']]['bndbox']),
                        'predicate': rel['predicate'],
                        'object': self._denoise_names(
                            object_dict[rel['object_id']]['name']),
                        'object_box': self._decode_box(
                            object_dict[rel['object_id']]['bndbox'])
                    }
                    for rel in anno['relationships']
                ]
            }
            for anno, object_dict in zip(annos, objects)
            if anno['filename'].split('.')[0] not in corrupted_images
            and heights_widths[anno['filename']][0] is not None
            and any(anno['relationships'])
        ]

    def download_annotations(self):
        """Download VrR-VG annotations."""
        if not os.path.exists(self._orig_annos_path):
            os.mkdir(self._orig_annos_path)
        gdown.download(
            'https://drive.google.com/uc?id=15i5J87FHfZ1DwPkumBaH50nfOg01Ja8C',
            self._orig_annos_path + 'VrR-VG.zip', quiet=False
        )
        with ZipFile(self._orig_annos_path + 'VrR-VG.zip') as fid:
            fid.extractall(self._orig_annos_path)
        os.remove(self._orig_annos_path + 'VrR-VG.zip')
        os.system("wget http://svl.stanford.edu/projects/scene-graph/VG/image_data.json")
        os.system("wget http://svl.stanford.edu/projects/scene-graph/dataset/VG-SGG.h5")
        shutil.move(
            'image_data.json',
            self._orig_annos_path + 'image_data.json'
        )
        shutil.move('VG-SGG.h5', self._orig_annos_path + 'VG-SGG.h5')

    @staticmethod
    def _decode_box(box):
        return [
            int(box['ymin']), int(box['ymax']),
            int(box['xmin']), int(box['xmax'])
        ]

    def _denoise_names(self, name):
        name = name.replace("man's", "man")
        if name in self._noisy_names:
            return self._noisy_names[name]
        return name

    def _set_split_ids(self):
        # Load images' metadata
        with open(self._orig_annos_path + 'image_data.json') as fid:
            img_names = [
                img['url'].split('/')[-1]
                for img in json.load(fid)
                if img['image_id'] not in [1592, 1722, 4616, 4617]
            ]
        annos = h5py.File(self._orig_annos_path + 'VG-SGG.h5', 'r')
        self._split_ids = {
            name: split_id
            for name, split_id in zip(img_names, annos['split'][:])
        }
        for name, split_id in self._split_ids.items():
            if split_id == 0 and random.uniform(0, 1) < 0.02:
                self._split_ids[name] = 1

    def _xml_to_dict(self, node):
        annos = {
            child.tag: (
                self._xml_to_dict(child) if child.text is None else child.text
            )
            for child in node
            if child.tag not in ('object', 'relation')
        }
        if any(child.tag == 'object' for child in node):
            annos['objects'] = [
                self._xml_to_dict(child)
                for child in node if child.tag == 'object'
            ]
        if any(child.tag == 'relation' for child in node):
            annos['relationships'] = [
                self._xml_to_dict(child)
                for child in node if child.tag == 'relation'
            ]
        return annos
