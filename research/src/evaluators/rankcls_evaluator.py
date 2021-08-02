# -*- coding: utf-8 -*-
"""Class to compute accuracy metrics for bg/fg classification."""

import numpy as np
from sklearn.metrics import f1_score


class RankingClsEvaluator:
    """A class providing methods to evaluate ranking (bg vs fg)."""

    def __init__(self, annotation_loader):
        """Initialize evaluator setup for this dataset."""
        self.reset()

        # Ground-truth labels and boxes
        annos = annotation_loader.get_annos()
        self._annos = {
            anno['filename']: anno for anno in annos if anno['split_id'] == 2}

    def reset(self):
        """Initialize counters."""
        self._gt_positive_counter = []
        self._true_positive_counter = []
        self._f1_scores = []

    def step(self, filename, scores):
        """
        Evaluate accuracy for a given image.

        Inputs:
            - filename: str, name of the image to evaluate
            - scores: array (n_det, n_classes)
        """
        # Update true positive counter and get gt labels-bboxes
        if filename in self._annos.keys():
            self._gt_positive_counter.append(
                len(self._annos[filename]['relations']['ids']))
            gt_labels = np.zeros(
                self._annos[filename]['relations']['ids'].shape)
            gt_names = self._annos[filename]['relations']['names']
            gt_labels[gt_names == "__background__"] = 0
            gt_labels[gt_names != "__background__"] = 1

        # Compute the different recall types
        if filename in self._annos.keys():
            det_classes = np.argsort(scores)[:, ::-1]
            keep_top_1 = det_classes[:, 0] == gt_labels
            self._true_positive_counter.append(
                len(det_classes[keep_top_1]))
            self._f1_scores.append(f1_score(gt_labels, det_classes[:, 0]))

    def print_stats(self):
        """Print accuracy statistics."""
        for rmode in ('micro', 'macro'):
            print('Ranking %sAccuracy:' % (rmode), self._compute_acc(rmode))
        print('Ranking f1-score:', 100 * np.mean(self._f1_scores))

    def _compute_acc(self, rmode):
        """Compute micro or macro accuracy."""
        if rmode == 'micro':
            return (  # sum over tp / sum over gt
                100 * np.sum(self._true_positive_counter, axis=0)
                / np.sum(self._gt_positive_counter))
        return (  # mean over (tp_i / gt_i) for each image i
            100 * np.mean(
                np.array(self._true_positive_counter)
                / np.array(self._gt_positive_counter),
                axis=0))
