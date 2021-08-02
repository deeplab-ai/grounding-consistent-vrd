# -*- coding: utf-8 -*-
"""Class to compute precision metrics for relationship detection."""

import pickle
import json

import numpy as np


R1_PREDS = {'VRD': {'carry', 'contain', 'cover', 'drive', 'eat', 'feed', 'fly', 'has', 'hit',
                    'hold', 'kick', 'play with', 'pull', 'ride', 'touch', 'use', 'wear', 'with'},
            'VG200': {'carrying', 'eating', 'has', 'holding', 'playing', 'riding', 'using',
                      'wearing', 'wears', 'with'}
            }

R2_PREDS = {'VRD': {'at', 'drive on', 'in', 'inside', 'lean on', 'lying on', 'on', 'park on',
                    'rest on', 'sit on', 'skate on', 'sleep on', 'stand on'},
            'VG200': {'at', 'attached to', 'belonging to', 'flying in', 'for', 'from',
                      'growing on', 'hanging from', 'in', 'laying on', 'looking at',
                      'lying on', 'made of', 'mounted on', 'of', 'on', 'painted on',
                      'parked on', 'part of', 'says', 'sitting on', 'standing on', 'to',
                      'walking in', 'walking on', 'watching'}
            }


def get_proximals(dataset):
    return R1_PREDS[dataset].union(R2_PREDS[dataset])


class NegativesEvaluator:
    """A class providing methods to evaluate precision."""

    def __init__(self, annotation_loader, use_merged=False, typ='fmP+'):
        """Initialize evaluator setup for this dataset."""
        self.dataset = annotation_loader._dataset
        self.typ = typ
        if typ.startswith('f'):
            self.PREDICATES = get_proximals(self.dataset)
        else:
            with open(annotation_loader._json_path + self.dataset + '_predicates.json') as fid:
                self.PREDICATES = np.array(json.load(fid))
        self.reset()

        # Ground-truth labels and boxes
        annotation_loader.reset(annotation_loader._mode)
        annos = annotation_loader.get_annos()
        if use_merged:
            for anno in annos:
                anno['relations']['ids'] = anno['relations']['merged_ids']
        self._annos = {
            anno['filename']: anno
            for anno in annos if anno['split_id'] == 2
        }
        self.pred2id = {
            name: _id
            for anno in self._annos.values()
            for name, _id in
            zip(anno['relations']['names'], anno['relations']['ids'])
        }
        self.bg_idx = self.pred2id['__background__']
        # Connections to classes after merge
        self._connections = None
        if use_merged:
            dataset = annotation_loader._dataset
            json_path = annotation_loader._json_path
            with open(json_path + dataset + '_merged.pkl', 'rb') as fid:
                self._connections = pickle.load(fid)

    def reset(self):
        """Initialize counters."""
        self._true_positive_counter = {pr: 0 for pr in self.PREDICATES}
        self._positive_counter = {pr: 0 for pr in self.PREDICATES}
        self._sample_counter = {pr: 0 for pr in self.PREDICATES}

    def step(self, filename, scores, labels=None, boxes=None, phr_rec=None):
        """
        Evaluate accuracy for a given image.

        Inputs:
            - filename: str, name of the image to evaluate
            - scores: array (n_det, n_classes)
        """
        if filename in self._annos.keys():
            gt_labels = np.array(self._annos[filename]['relations']['ids'])
            subj_ids = self._annos[filename]['objects']['ids'][
                self._annos[filename]['relations']['subj_ids']
            ]
            obj_ids = self._annos[filename]['objects']['ids'][
                self._annos[filename]['relations']['obj_ids']
            ]
            neg_ids = np.array(self._annos[filename]['relations']['neg_ids'])
            det_classes = np.argsort(scores)[:, ::-1][:, 0]
            if self._connections is not None:
                det_classes = np.array([
                    self._connections[subj, obj][pred]
                    for subj, pred, obj in zip(subj_ids, det_classes, obj_ids)
                ])
            for pred_name in self._true_positive_counter:
                if pred_name not in self.pred2id:
                    continue
                pred = self.pred2id[pred_name]
                # keep only labeled samples or negatives for pred
                if self.typ.endswith('+'):
                    negs = np.array([pred in sample for sample in neg_ids])
                else:
                    negs = np.array([False for sample in neg_ids])
                inds = (gt_labels != self.bg_idx)  | negs
                # tmp_gt: 1 if gt else 0
                tmp_gt = np.copy(gt_labels)[inds]
                tmp_gt[tmp_gt == pred] = -1
                tmp_gt[tmp_gt != -1] = 0
                tmp_gt = -tmp_gt  # change -1 to 1
                # tmp_det: 1 if gt else 0
                tmp_det = np.copy(det_classes)[inds]
                tmp_det[tmp_det == pred] = -1
                tmp_det[tmp_det != -1] = 0
                tmp_det = -tmp_det  # change -1 to 1
                # Positives
                pos = tmp_det.sum()
                # True positives
                tps = (tmp_det * tmp_gt).sum()
                # Labels of interest (positives and negatives for this class)
                lois = (gt_labels == pred).sum() + negs.sum()
                self._true_positive_counter[pred_name] += tps
                self._positive_counter[pred_name] += pos
                self._sample_counter[pred_name] += lois

    def print_stats(self, task=None):
        """Print accuracy statistics."""
        for rmode in ('micro', 'macro'):
            # for tmode in self._true_positive_counter:
            print('{}Precision: {:.2f}'.format(
                rmode, self._compute_precision(rmode, prnt=False)))

    def _compute_precision(self, rmode, prnt=False):
        """Compute micro or macro precision of typ group."""
        preds = sorted([
            key for key in self._sample_counter if self._sample_counter[key]
        ])
        precision_per_class = np.array([
            self._true_positive_counter[pred] / self._positive_counter[pred]
            if self._positive_counter[pred]
            else 0
            for pred in preds
        ])
        if prnt:
            for pred, prec in zip(preds, precision_per_class):
                print('{} {:.2f} {}'.format(pred, prec, self._sample_counter[pred]))
        if rmode == 'micro':
            weights = np.array([self._sample_counter[pred] for pred in preds])
        else:
            weights = np.ones((len(preds),))
        return 100 * np.sum(weights * precision_per_class) / np.sum(weights)
