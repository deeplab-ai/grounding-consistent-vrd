# -*- coding: utf-8 -*-
"""Class to compute accuracy metrics for relationship grounding."""

import numpy as np
import torch
import torch.nn.functional as F


class GroundEvaluator:
    """A class providing methods to evaluate grounding."""

    def __init__(self, annotation_loader, hmap_threshold=0.5):
        """Initialize evaluator setup for this dataset."""
        self.hmap_threshold = hmap_threshold
        self.reset()

        # Ground-truth labels and boxes
        annos = annotation_loader.get_annos()
        self._annos = {
            anno['filename']: anno for anno in annos if anno['split_id'] == 2
        }

    def reset(self):
        """Initialize counters."""
        self._gt_positive_counter = {'subj': [], 'obj': [], 'total': []}
        self._true_positive_counter = {'subj': [], 'obj': [], 'total': []}
        self._positive_counter = {'subj': [], 'obj': [], 'total': []}
        self._wmIoU = {'subj': [], 'obj': [], 'total': []}

    def step(self, filename, pred_masks, pred_nrm_boxes, gt_masks):
        """
        Evaluate accuracy for a given image.

        Inputs:
            - filename: str, name of the image to evaluate
            - pred_masks: [nparray, nparray], numpy arrays of subject/object
                predicted heatmaps, at least ones must not be None
            - pred_nrm_boxes: [nparray, nparray], numpy arrays of normalized
                width/2 and height/2
            - gt_masks: same as pred_masks, gt masks of subjects/objects
        """
        pred_masks = [
            pred_masks[0] / (pred_masks[0].flatten(1).sum(1) + 1e-8).view(-1, 1, 1),
            pred_masks[1] / (pred_masks[1].flatten(1).sum(1) + 1e-8).view(-1, 1, 1),
        ]
        pred_masks_dict = {'subj': pred_masks[0], 'obj': pred_masks[1]}
        pred_nrm_box_dict = {'subj': pred_nrm_boxes[0],
                             'obj': pred_nrm_boxes[1]}
        gt_masks_dict = {'subj': gt_masks[0].squeeze(1),
                         'obj': gt_masks[1].squeeze(1)}

        # Update true positive counter and get gt labels-bboxes
        if pred_masks[0] is not None and pred_masks[1] is not None:
            pred_masks_dict['total'] = torch.cat(
                (pred_masks[0], pred_masks[1]), dim=0).squeeze(1)
            gt_masks_dict['total'] = torch.cat(
                (gt_masks[0], gt_masks[1]), dim=0).squeeze(1)
            pred_nrm_box_dict['total'] = torch.cat(
                (pred_nrm_boxes[0], pred_nrm_boxes[1]), dim=0).squeeze(1)
        elif pred_masks[0] is not None:
            pred_masks_dict['total'] = pred_masks[0]
            gt_masks_dict['total'] = gt_masks[0]
            pred_nrm_box_dict['total'] = pred_nrm_boxes[0]
        elif pred_masks[1] is not None:
            pred_masks_dict['total'] = pred_masks[1]
            gt_masks_dict['total'] = gt_masks[1]
            pred_nrm_box_dict['total'] = pred_nrm_boxes[1]
        for typ in ['subj', 'obj', 'total']:
            if pred_masks_dict[typ] is None:
                continue
            shape = pred_masks_dict[typ].unsqueeze(0).shape
            gt_masks = gt_masks_dict[typ]
            norm_boxes = torch.clamp(pred_nrm_box_dict[typ], 0.0, 0.49)
            kerns = torch.zeros_like(pred_masks_dict[typ]).unsqueeze(1)
            for i in range(shape[1]):
                c = shape[-1] // 2
                w = torch.round(norm_boxes[i, 0] * shape[-2]
                                ).type(torch.long).item()
                h = torch.round(norm_boxes[i, 1] * shape[-1]
                                ).type(torch.long).item()
                kerns[i, 0, c-h:c+h+1, c-w:c+w+1] = 1
            # calculate intersections
            inters = F.conv2d(
                gt_masks.unsqueeze(0), kerns, padding=shape[-1]//2,
                groups=shape[1]).squeeze(0)
            # union = area1 + area2 - intersection
            gt_areas = gt_masks.flatten(1).sum(1)
            kern_areas = kerns.flatten(1).sum(1)
            unions = (gt_areas + kern_areas).view(-1, 1, 1).expand(
                -1, shape[-2], shape[-1]) - inters
            IoU = inters / unions
            # weighted mean IoU
            wmIoU = (IoU * pred_masks_dict[typ]).sum(-1).sum(-1)

            if filename in self._annos.keys():
                self._wmIoU[typ].append(wmIoU.cpu().numpy())


    def print_stats(self):
        """Print statistics."""
        for rmode in ('micro', 'macro'):
            print('{}Recall{{subj|obj|total}} {}  {}  {}'.format(
                rmode, *(wrap_dec(self._compute_wmIoU(rmode, typ))
                         for typ in ['subj', 'obj', 'total'])))

    def _compute_precision(self, rmode, typ):
        """Compute micro or macro precision of typ group."""
        if len(self._true_positive_counter[typ]) == 0:
            return '---'
        if rmode == 'micro':
            return (  # sum over tp / sum over gt
                100 * np.sum(self._true_positive_counter[typ], axis=0)
                / np.sum(self._positive_counter[typ]))
        return (  # mean over (tp_i / gt_i) for each image i
            100 * np.mean(
                np.array(self._true_positive_counter[typ])
                / np.array(self._positive_counter[typ]),
                axis=0))

    def _compute_wmIoU(self, rmode, typ):
        """Compute micro or macro accuracy."""
        if len(self._wmIoU[typ]) == 0:
            return '---'
        if rmode == 'micro':
            return 100 * np.concatenate(self._wmIoU[typ]).mean()
        return 100 * np.mean([u.mean() for u in self._wmIoU[typ]])

    def _compute_recall(self, rmode, typ):
        """Compute micro or macro accuracy."""
        if len(self._true_positive_counter[typ]) == 0:
            return '---'
        if rmode == 'micro':
            return (  # sum over tp / sum over gt
                100 * np.sum(self._true_positive_counter[typ], axis=0)
                / np.sum(self._gt_positive_counter[typ]))
        return (  # mean over (tp_i / gt_i) for each image i
            100 * np.mean(
                np.array(self._true_positive_counter[typ])
                / np.array(self._gt_positive_counter[typ]),
                axis=0))


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


def wrap_dec(x):
    """Allow two floating points and strings"""
    if isinstance(x, str):
        return x
    return '{:.2f}'.format(x)
