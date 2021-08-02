# -*- coding: utf-8 -*-
"""Class to compute accuracy metrics for object classification."""

import numpy as np


class ObjectClsEvaluator:
    """A class providing methods to evaluate the ObjCls problem."""

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
        self._true_positive_counter = {'top-1': [], 'top-5': []}

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
                len(self._annos[filename]['objects']['ids']))
            gt_labels = self._annos[filename]['objects']['ids']

        # Compute the different recall types
        if filename in self._annos.keys():
            det_classes = np.argsort(scores)[:, ::-1]
            keep_top_1 = det_classes[:, 0] == gt_labels
            keep_top_5 = (det_classes[:, :5] == gt_labels[:, None]).any(1)
            self._true_positive_counter['top-1'].append(
                len(det_classes[keep_top_1]))
            self._true_positive_counter['top-5'].append(
                len(det_classes[keep_top_5]))

    def print_stats(self):
        """Print accuracy statistics."""
        for rmode in ('micro', 'macro'):
            for tmode in ('top-1', 'top-5'):
                print(
                    '%sAccuracy %s:'
                    % (rmode, tmode),
                    self._compute_acc(rmode, tmode)
                )

    def _compute_acc(self, rmode, tmode):
        """Compute micro or macro accuracy."""
        if rmode == 'micro':
            return (  # sum over tp / sum over gt
                100 * np.sum(
                    self._true_positive_counter[tmode],
                    axis=0)
                / np.sum(self._gt_positive_counter))
        return (  # mean over (tp_i / gt_i) for each image i
            100 * np.mean(
                np.array(self._true_positive_counter[tmode])
                / np.array(self._gt_positive_counter),
                axis=0))
