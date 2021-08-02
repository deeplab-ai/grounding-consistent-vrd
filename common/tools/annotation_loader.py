# -*- coding: utf-8 -*-
"""A custom loader of annotations using multiple filters."""

from collections import Counter
from copy import deepcopy
import json
import os
from random import random

import numpy as np


class AnnotationLoader:
    """A class to load and filter annotations."""

    def __init__(self, config):
        """Initialize loader."""
        self.config = config
        self._set_from_config(config)
        self.reset()

    def _set_from_config(self, config):
        """Load config variables."""
        self._bg_perc = config.bg_perc
        self._dataset = config.dataset
        self._classes_to_keep = config.classes_to_keep
        self._filter_duplicate_rels = config.filter_duplicate_rels
        self._filter_multiple_preds = config.filter_multiple_preds
        self._json_path = config.paths['json_path']
        self._mode = config.task
        self._orig_images_path = config.orig_img_path
        self._pairs_limit = config.relations_per_img_limit
        self._train_with_negatives = config.use_negative_samples
        self._test_with_negatives = config.test_on_negatives

    def reset(self, mode=None):
        """Reset loader with new mode."""
        self._annos = []
        if mode is not None:
            self._mode = mode

    def get_annos(self):
        """Return full filtered annotations."""
        if not self._annos:
            self._annos = self._load_annotations()
            if not self._mode.startswith('obj'):
                self._annos = self._filter_annotations(self._annos)
        return self._annos

    def get_class_counts(self, feature='relations'):
        """Return class frequencies for relations or objects."""
        annos = self.get_annos()
        cntr = Counter([_id for anno in annos for _id in anno[feature]['ids']])
        return np.array([cntr[_id] for _id in sorted(list(cntr.keys()))])

    def get_zs_annos(self):
        """Return zero-shot annotations."""
        if not self._annos:
            self._annos = self._load_annotations()
            self._annos = self._filter_annotations(self._annos)
        seen = set(
            (anno['objects']['ids'][sid], rel_id, anno['objects']['ids'][oid])
            for anno in self._annos if anno['split_id'] == 0
            for sid, rel_id, oid in zip(
                anno['relations']['subj_ids'],
                anno['relations']['ids'],
                anno['relations']['obj_ids']
            )
        )
        zs_annos = []
        for anno in self._annos:
            if anno['split_id'] == 2 and anno['relations']['names'].tolist():
                keep = [
                    r for r, (sid, rid, oid) in enumerate(zip(
                        anno['objects']['ids'][anno['relations']['subj_ids']],
                        anno['relations']['ids'],
                        anno['objects']['ids'][anno['relations']['obj_ids']]
                    ))
                    if (sid, rid, oid) not in seen
                ]
                if keep:
                    anno = deepcopy(anno)
                    anno['relations'] = {
                        'ids': anno['relations']['ids'][keep],
                        'merged_ids': anno['relations']['merged_ids'][keep],
                        'names': anno['relations']['names'][keep],
                        'subj_ids': anno['relations']['subj_ids'][keep],
                        'obj_ids': anno['relations']['obj_ids'][keep]
                    }
                    zs_annos.append(anno)
        return zs_annos

    def _filter_annotations(self, annotations):
        """Apply specified filters on annotations."""
        # Enhance with negatives
        annotations = self._merge_negatives(annotations)
        # Ensure there are foreground samples if preddet, no bg in eval
        if self._mode == 'preddet':
            annotations = [
                self._filter_bg(anno) if anno['split_id'] != 0 else anno
                for anno in annotations
                if len(set(anno['relations']['names'].tolist())) > 1
            ]
        # Keep only samples of specific classes
        if self._classes_to_keep is not None:
            for anno in annotations:
                if anno['split_id'] == 2:
                    anno['relations'] = self._filter_nontail(anno['relations'])
        annotations = [
            anno for anno in annotations if anno['relations']['names'].tolist()
        ]
        # Filter duplicate triplets per pair
        if self._filter_duplicate_rels:
            for anno in annotations:
                if anno['split_id'] == 0:
                    anno['relations'] = self._filter_dupls(anno['relations'])
        # Filter multiple triplets per pair
        if self._filter_multiple_preds:
            for anno in annotations:
                if anno['split_id'] == 0:
                    anno['relations'] = self._filter_multi(anno['relations'])
        # Keep at most relations_per_img_limit pairs for memory issues
        for anno in annotations:
            if anno['split_id'] == 0:
                anno['relations'] = self._filter_pairs(anno['relations'])
        return annotations

    @staticmethod
    def _filter_bg(anno):
        """Filter background annotations."""
        inds = np.array([
            n for n, name in enumerate(anno['relations']['names'])
            if name != '__background__' or anno['relations']['neg_ids'][n]
        ])
        anno['relations']['names'] = anno['relations']['names'][inds]
        anno['relations']['ids'] = anno['relations']['ids'][inds]
        anno['relations']['merged_ids'] = anno['relations']['merged_ids'][inds]
        anno['relations']['subj_ids'] = anno['relations']['subj_ids'][inds]
        anno['relations']['obj_ids'] = anno['relations']['obj_ids'][inds]
        anno['relations']['neg_ids'] = anno['relations']['neg_ids'][inds]
        return anno

    @staticmethod
    def _filter_dupls(relations):
        """Filter relations appearing more than once."""
        _, unique_inds = np.unique(np.stack(
            (relations['subj_ids'], relations['ids'], relations['obj_ids']),
            axis=1
        ), axis=0, return_index=True)
        return {
            'ids': relations['ids'][unique_inds],
            'merged_ids': relations['merged_ids'][unique_inds],
            'names': relations['names'][unique_inds],
            'subj_ids': relations['subj_ids'][unique_inds],
            'obj_ids': relations['obj_ids'][unique_inds],
            'neg_ids': relations['neg_ids'][unique_inds]
        }

    @staticmethod
    def _filter_multi(relations):
        """Filter multiple annotations for the same object pair."""
        _, unique_inds = np.unique(np.stack(
            (relations['subj_ids'], relations['obj_ids']), axis=1
        ), axis=0, return_index=True)
        return {
            'ids': relations['ids'][unique_inds],
            'merged_ids': relations['merged_ids'][unique_inds],
            'names': relations['names'][unique_inds],
            'subj_ids': relations['subj_ids'][unique_inds],
            'obj_ids': relations['obj_ids'][unique_inds],
            'neg_ids': relations['neg_ids'][unique_inds]
        }

    def _filter_nontail(self, relations):
        """Filter non-tail classes."""
        keep_inds = relations['ids'][:, None] == self._classes_to_keep[None, :]
        keep_inds = keep_inds.any(1)
        return {
            'ids': relations['ids'][keep_inds],
            'merged_ids': relations['merged_ids'][keep_inds],
            'names': relations['names'][keep_inds],
            'subj_ids': relations['subj_ids'][keep_inds],
            'obj_ids': relations['obj_ids'][keep_inds],
            'neg_ids': relations['neg_ids'][keep_inds]
        }

    def _filter_pairs(self, relations):
        """Limit pairs per image for memory issues."""
        relations['names'] = relations['names'][:self._pairs_limit]
        relations['ids'] = relations['ids'][:self._pairs_limit]
        relations['merged_ids'] = relations['merged_ids'][:self._pairs_limit]
        relations['subj_ids'] = relations['subj_ids'][:self._pairs_limit]
        relations['obj_ids'] = relations['obj_ids'][:self._pairs_limit]
        relations['neg_ids'] = relations['neg_ids'][:self._pairs_limit]
        return relations

    def _load_annotations(self):
        """Load annotations from json."""
        _mode = '_sggen_merged' if self._mode == 'sggen' else '_predcls'
        if self._dataset in {'VG80K', 'VrR-VG'}:
            _mode = '_preddet'
        with open(self._json_path + self._dataset + _mode + '.json') as fid:
            annotations = json.load(fid)
        return self._to_list_with_arrays(annotations)

    def _merge_negatives(self, annotations):
        """Merge with negative labels."""
        negatives = {}
        if self._train_with_negatives or self._test_with_negatives:
            neg_json = self._json_path + self._dataset + '_negatives.json'
            with open(neg_json) as fid:
                negatives = json.load(fid)
        for anno in annotations:
            neg_ids = [[] for _ in range(len(anno['relations']['ids']))]
            update_neg_ids = (
                anno['filename'] in negatives
                and (
                    (anno['split_id'] < 2 and self._train_with_negatives)
                    or (anno['split_id'] == 2 and self._test_with_negatives)
                )
            )
            if update_neg_ids:
                neg_ids = negatives[anno['filename']]
            anno['relations']['neg_ids'] = np.copy(neg_ids)
        return annotations

    def _to_list_with_arrays(self, annotations):
        """Transform lists to numpy arrays."""
        orig_img_names = set(os.listdir(self._orig_images_path))
        if self._dataset == 'COCO':  # COCO training mines from VRD/VG images
            orig_img_names = orig_img_names.union(
                set(os.listdir(self._orig_images_path.replace('COCO', 'VRD')))
            )
            orig_img_names = orig_img_names.union(
                set(os.listdir(self._orig_images_path.replace('COCO', 'VG')))
            )
        return [
            {
                'filename': anno['filename'],
                'split_id': anno['split_id'],
                'height': anno['height'],
                'width': anno['width'],
                'dataset': anno['dataset'] if 'dataset' in anno else None,
                'objects': {
                    'boxes': np.array(anno['objects']['boxes']),
                    'ids': np.array(anno['objects']['ids']).astype(int),
                    'names': np.array(anno['objects']['names']),
                    'scores': (
                        np.array(anno['objects']['scores'])
                        if 'scores' in anno['objects']
                        and anno['objects']['scores'] is not None
                        else None)
                },
                'relations': {
                    'ids': np.array(anno['relations']['ids']).astype(int),
                    'merged_ids': np.array(
                        anno['relations']['merged_ids']).astype(int),
                    'names': np.array(anno['relations']['names']),
                    'subj_ids': np.array(anno['relations']['subj_ids']),
                    'obj_ids': np.array(anno['relations']['obj_ids'])
                }
            }
            for anno in annotations
            if anno['filename'] in orig_img_names
            and (any(anno['relations']['names']) or 'obj' in self._mode)
            and any(anno['objects']['names'])
            and not ('dataset' in anno and anno['dataset'] == 'COCO')
        ]
