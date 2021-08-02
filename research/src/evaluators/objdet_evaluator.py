# -*- coding: utf-8 -*-
"""Class to compute mAP average precision for object detection."""

from collections import defaultdict

import numpy as np


class ObjectDetEvaluator:
    """A class providing methods to evaluate the ObjDet problem."""

    def __init__(self, annotation_loader):
        """Initialize evaluator setup for this dataset."""
        self.reset()

        # Ground-truth labels and boxes
        annos = annotation_loader.get_annos()
        self._annos = {
            anno['filename']: anno for anno in annos if anno['split_id'] == 2}

    def reset(self):
        """Initialize positive counters."""
        self._gt_positives = defaultdict(int)
        self._true_positives = defaultdict(list)
        self._scores = defaultdict(list)

    def step(self, filename, scores, boxes, labels):
        """
        Evaluate the detections of a given image.

        Inputs:
            - filename: str, name of the image to evaluate
            - scores: array (n_det,)
            - boxes: array (n_det, 4)
            - labels: array (ndet,)
        """
        # Sort detections based on their scores
        score_sort = scores.argsort()[::-1]
        labels = labels[score_sort]
        boxes = boxes[score_sort]
        scores = scores[score_sort]

        # Get gt annotations and update gt counter
        if filename in self._annos.keys():
            gt_boxes = self._annos[filename]['objects']['boxes']
            gt_classes = self._annos[filename]['objects']['ids']
            for cid in gt_classes:
                self._gt_positives[cid] += 1

        # Compute the different recall types
        if filename in self._annos.keys():
            tps = detection_precision(labels, boxes, gt_classes, gt_boxes)
            for cid, score, tp_value in zip(labels, scores, tps):
                self._true_positives[cid].append(tp_value)
                self._scores[cid].append(score)

    def print_stats(self):
        """Print mAP statistics."""
        print('Mean Average Precision:', self._compute_map())

    def _compute_map(self):
        """Compute mean average precision."""
        for name in self._gt_positives:
            if name not in self._true_positives:
                self._true_positives[name] = [0]
                self._scores[name] = [0]
            score_sort = np.argsort(self._scores[name])[::-1]
            self._true_positives[name] = np.array(
                self._true_positives[name]
            )[score_sort]
        aps = [
            voc_ap(
                np.cumsum(self._true_positives[name])
                / self._gt_positives[name],
                np.cumsum(self._true_positives[name])
                / np.cumsum(np.ones_like(self._true_positives[name]))
            )
            for name in sorted(self._gt_positives.keys())
        ]
        return np.mean(aps) * 100


def compute_area(bbox):
    """Compute area of box 'bbox' ([y_min, y_max, x_min, x_max])."""
    return max(0, bbox[3] - bbox[2] + 1) * max(0, bbox[1] - bbox[0] + 1)


def compute_overlap(det_bboxes, gt_bboxes):
    """
    Compute overlap of detected and ground truth boxes.

    Inputs:
        - det_bboxes: array (n, 4), n x [y_min, y_max, x_min, x_max]
            The detected bounding boxes for subject and object
        - gt_bboxes: array (n, 4), n x [y_min, y_max, x_min, x_max]
            The ground truth bounding boxes for subject and object
        n is 2 in case of relationship recall, 1 in case of phrases
    Returns:
        - overlap: non-negative float <= 1
    """
    overlaps = []
    for det_bbox, gt_bbox in zip(det_bboxes, gt_bboxes):
        intersection_bbox = [
            max(det_bbox[0], gt_bbox[0]),
            min(det_bbox[1], gt_bbox[1]),
            max(det_bbox[2], gt_bbox[2]),
            min(det_bbox[3], gt_bbox[3])
        ]
        intersection_area = compute_area(intersection_bbox)
        union_area = (compute_area(det_bbox)
                      + compute_area(gt_bbox)
                      - intersection_area)
        overlaps.append(intersection_area / union_area)
    return min(overlaps)


def detection_precision(det_labels, det_bboxes, gt_labels, gt_bboxes,
                        min_overlap=0.5):
    """
    Evaluate precision, detecting true positives.

    Inputs:
        - det_labels: array (Ndet,) of detected labels,
            where Ndet is the number of predictions in this image
        - det_bboxes: array (Ndet, 4) of detected boxes,
            where Ndet is the number of predictions in this image and
            each 1x4 array: [y_min, y_max, x_min, x_max]
        - gt_labels: array (N,) of ground-truth labels
        - gt_bboxes: array (N, 4) of ground-truth boxes
        - min_overlap: float, overlap threshold to consider detection
    Returns:
        - a binary list with 1 indicating a true positive
    """
    # Check only detections that match any of the ground-truth
    possible_matches = det_labels[..., None] == gt_labels.T[None, ...]
    check_inds = possible_matches.any(1)
    true_positives = np.copy(check_inds) * 1
    for ind, bbox in zip(np.where(check_inds)[0], det_bboxes[check_inds]):
        overlaps = np.array([
            compute_overlap([bbox], [gt_box]) if match else 0
            for gt_box, match in zip(gt_bboxes, possible_matches[ind])
        ])
        if (overlaps >= min_overlap).any():
            possible_matches[:, np.argmax(overlaps)] = False
        else:
            true_positives[ind] = 0
    return true_positives.tolist()


def voc_ap(recall, precision):
    """Code to compute Average Precision as given by PASCAL VOC 2012."""
    rec = np.zeros(len(recall) + 2)
    rec[1:-1] = recall
    rec[-1] = 1.0
    prec = np.zeros(len(precision) + 2)
    prec[1:-1] = precision
    # Make the precision monotonically decreasing
    for i in range(len(prec) - 2, -1, -1):
        prec[i] = max(prec[i], prec[i + 1])
    # Return the area under the curve (numerical integration)
    return np.sum((rec[1:] - rec[:-1]) * prec[1:])
