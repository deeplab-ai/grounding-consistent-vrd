# -*- coding: utf-8 -*-
"""Transform annotations into a standard desired format."""

import json
import os
from zipfile import ZipFile

import gdown

from .dataset_transformer_class import DatasetTransformer


class SVGTransformer(DatasetTransformer):
    """Extands DatasetTransformer for sVG."""

    def __init__(self, config):
        """Initialize SVGTransformer."""
        super().__init__(config)
        self.r1_preds = {
            'cut', 'eat', 'enjoy', 'have', 'hold', 'play', 'read', 'ride',
            'wear', 'with'
        }
        self.r2_preds = {
            'at', 'attached to', 'hang on', 'in', 'of', 'on', 'watch'
        }

    def create_relationship_json(self):
        """Transform sVG annotations."""
        with open(self._orig_annos_path + 'svg_train.json') as fid:
            annos = json.load(fid)
        json_annos = self._merge_rel_annos(annos, 0)
        for anno in json_annos[-2000:]:
            anno['split_id'] = 1
        with open(self._orig_annos_path + 'svg_test.json') as fid:
            annos = json.load(fid)
        json_annos += self._merge_rel_annos(annos, 2)
        noisy_images = {  # contain bboxes > image
            '1191.jpg', '1360.jpg', '1159.jpg',
            '1018.jpg', '1327.jpg', '1280.jpg'
        }
        json_annos = [
            anno for anno in json_annos if anno['filename'] not in noisy_images
        ]
        return self._clear_obj_annos(json_annos)

    def download_annotations(self):
        """Download sVG annotations."""
        if not os.path.exists(self._orig_annos_path):
            os.mkdir(self._orig_annos_path)
        orig_files = ['svg_train.json', 'svg_test.json']
        orig_files = [self._orig_annos_path + name for name in orig_files]
        if not all(os.path.exists(name) for name in orig_files):
            gdown.download(
                'https://drive.google.com/uc?id=0B5RJWjAhdT04SXRfVHBKZ0dOTzQ',
                self._orig_annos_path + 'svg.zip', quiet=False
            )
            with ZipFile(self._orig_annos_path + 'svg.zip') as fid:
                fid.extractall(self._orig_annos_path)
            os.remove(self._orig_annos_path + 'svg.zip')

    @staticmethod
    def _clear_obj_annos(annos):
        noisy_words = {
            "streetsign": "street sign",
            "theoutdoors": "outdoors",
            "licenseplate": "license plate",
            "stopsign": "stop sign",
            "toiletpaper": "toilet paper",
            "tennisracket": "tennis racket",
            "treetrunk": "tree trunk",
            "trafficlight": "traffic light",
            "bluesky": "blue sky",
            "firehydrant": "fire hydrant",
            "t-shirt": "t_shirt",
            "whiteclouds": "white clouds",
            "traincar": "train car",
            "tennisplayer": "tennis player",
            "skipole": "ski pole",
            "tenniscourt": "tennis court",
            "tennisball": "tennis ball",
            "baseballplayer": "baseball player"
        }
        for anno in annos:
            for rel in anno['relationships']:
                if rel['subject'] in noisy_words:
                    rel['subject'] = noisy_words[rel['subject']]
                if rel['object'] in noisy_words:
                    rel['object'] = noisy_words[rel['object']]
        return annos

    def _merge_rel_annos(self, annos, split_id):
        heights_widths = [
            self._compute_im_size(anno['url'].split('/')[-1])
            for anno in annos
        ]
        return [
            {
                'filename': anno['url'].split('/')[-1],
                'split_id': split_id,
                'height': shw[0],
                'width': shw[1],
                'relationships': [
                    {
                        'subject': rel['phrase'][0],
                        'subject_box': [
                            rel['subject'][1], rel['subject'][3],
                            rel['subject'][0], rel['subject'][2]],
                        'predicate': rel['phrase'][1],
                        'object': rel['phrase'][2],
                        'object_box': [
                            rel['object'][1], rel['object'][3],
                            rel['object'][0], rel['object'][2]]
                    }
                    for rel in anno['relationships']
                ]
            }
            for anno, shw in zip(annos, heights_widths)
            if shw[0] is not None
        ]
