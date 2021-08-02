# -*- coding: utf-8 -*-
"""Class to compute recall metrics for VRD-SGGen."""

import pickle

import numpy as np


class RelationshipEvaluator:
    """A class providing methods to evaluate the VRD-SGGen problem."""

    def __init__(self, annotation_loader, use_merged=False):
        """Initialize evaluator setup for this dataset."""
        self._recall_types = np.array([20, 50, 100])  # R@20, 50, 100
        self._max_recall = self._recall_types[-1]
        self.reset()

        # Ground-truth labels and boxes
        annotation_loader.reset('preddet')
        annos = annotation_loader.get_annos()
        if use_merged:
            for anno in annos:
                anno['relations']['ids'] = anno['relations']['merged_ids']
        zeroshot_annos = annotation_loader.get_zs_annos()
        if use_merged:
            for anno in zeroshot_annos:
                anno['relations']['ids'] = anno['relations']['merged_ids']
        self._annos = {
            'full': {
                anno['filename']: anno
                for anno in annos if anno['split_id'] == 2},
            'zeroshot': {anno['filename']: anno for anno in zeroshot_annos}
        }
        # Connections to classes after merge
        self._connections = None
        if use_merged:
            dataset = annotation_loader._dataset
            json_path = annotation_loader._json_path
            with open(json_path + dataset + '_merged.pkl', 'rb') as fid:
                self._connections = pickle.load(fid)

    def reset(self):
        """Initialize recall_counters."""
        self._gt_positive_counter = {'full': [], 'zeroshot': []}
        self._true_positive_counter = {
            (rmode, cmode, dmode): []
            for rmode in ('micro', 'macro')
            for cmode in ('graph constraints', 'no constraints')
            for dmode in ('full', 'zeroshot')
        }

    def step(self, filename, scores, labels, boxes, phrase_recall):
        """
        Evaluate relationship or phrase recall.

        Inputs:
            - filename: str, name of the image to evaluate
            - scores: array (n_det, n_classes)
            - labels: array (n_det, 3): [subj_cls, -1, obj_cls]
            - boxes: array (n_det, 2, 4)
            - phrase_recall: bool, whether to evaluate phrase recall
        """
        # Update true positive counter and get gt labels-bboxes
        gt_labels, gt_bboxes = {}, {}
        for dmode in ('full', 'zeroshot'):  # data mode
            if filename not in self._annos[dmode].keys():
                continue
            self._gt_positive_counter[dmode].append(
                len(self._annos[dmode][filename]['relations']['ids']))
            gt_labels[dmode], gt_bboxes[dmode] = self._get_gt(filename, dmode)

        # Compute the different recall types
        for rmode in ('micro', 'macro'):  # recall mode
            for cmode in ('graph constraints', 'no constraints'):  # constraint
                if filename in self._annos['full'].keys():
                    det_labels, det_bboxes = self._sort_detected(
                        scores, boxes, labels,
                        cmode == 'graph constraints', rmode == 'macro')
                for dmode in ('full', 'zeroshot'):  # data mode
                    if filename not in self._annos[dmode].keys():
                        continue
                    self._true_positive_counter[(rmode, cmode, dmode)].append(
                        relationship_recall(
                            self._recall_types, det_labels, det_bboxes,
                            gt_labels[dmode],
                            gt_bboxes[dmode],
                            macro_recall=rmode == 'macro',
                            phrase_recall=phrase_recall))

    def print_stats(self, task):
        """Print recall statistics for given task."""
        for rmode in ('micro', 'macro'):
            for cmode in ('graph constraints', 'no constraints'):
                for dmode in ('full', 'zeroshot'):
                    print(
                        '%sRecall@20-50-100 %s %s with %s:'
                        % (rmode, task, dmode, cmode),
                        self._compute_recall(rmode, cmode, dmode)
                    )

    def _compute_recall(self, rmode, cmode, dmode):
        """Compute micro or macro recall."""
        if rmode == 'micro':
            return (  # sum over tp / sum over gt
                100 * np.sum(
                    self._true_positive_counter[(rmode, cmode, dmode)],
                    axis=0)
                / np.sum(self._gt_positive_counter[dmode]))
        return (  # mean over (tp_i / gt_i) for each image i
            100 * np.mean(
                np.array(self._true_positive_counter[(rmode, cmode, dmode)])
                / np.array(self._gt_positive_counter[dmode])[:, None],
                axis=0))

    def _get_gt(self, filename, dmode):
        """
        Return ground truth labels and bounding boxes.

        - gt_labels: array (n_gt, 3), (subj, pred, obj)
        - gt_bboxes: array (n_t, 2, 4) (subj.-obj. boxes)
        """
        anno = self._annos[dmode][filename]
        gt_labels = np.stack((
            anno['objects']['ids'][anno['relations']['subj_ids']],
            anno['relations']['ids'],
            anno['objects']['ids'][anno['relations']['obj_ids']]
        ), axis=1)
        gt_bboxes = np.stack((
            anno['objects']['boxes'][anno['relations']['subj_ids']],
            anno['objects']['boxes'][anno['relations']['obj_ids']]
        ), axis=1)
        return gt_labels, gt_bboxes

    def _sort_detected(self, scores, boxes, labels,
                       graph_constraints=True, macro_recall=False):
        """
        Merge detected scores, labels and boxes to desired format.

        Inputs:
            - scores: array (n_det, n_classes)
            - boxes: array (n_det, 2, 4)
            - labels: array (n_det, 3): [subj_cls, -1, obj_cls]
            - graph_constraints: bool, when False, evaluate multilabel
            - macro_recall: bool, when True, clear duplicate detections
        Returns:
            - det_labels: array (N, 3), [subj_id, pred_id, obj_id]
            - det_bboxes: array (N, 2, 4), [subj_box, obj_box]
        """
        if macro_recall:  # clear duplicate detections
            _, unique_dets = np.unique(
                np.concatenate((labels, boxes.reshape(-1, 8), scores), axis=1),
                axis=0, return_index=True)
            scores = scores[unique_dets]
            boxes = boxes[unique_dets]
            labels = labels[unique_dets]

        scores = scores[:, :-1]  # clear background scores
        # Sort scores of each pair
        classes = np.argsort(scores)[:, ::-1]
        scores = np.sort(scores)[:, ::-1]
        if graph_constraints:  # only one prediction per pair
            classes = classes[:, :1]
            scores = scores[:, :1]

        # Sort across image and keep top-100 predictions
        top_detections_indices = np.unravel_index(
            np.argsort(scores, axis=None)[::-1][:self._max_recall],
            scores.shape)
        det_labels = labels[top_detections_indices[0]]
        det_labels[:, 1] = classes[top_detections_indices]
        det_bboxes = boxes[top_detections_indices[0]]
        # If merged evaluation, transform labels
        if self._connections is not None:
            det_labels[:, 1] = np.array([
                self._connections[label[0], label[2]][int(label[1])]
                for label in det_labels
            ])
        return det_labels, det_bboxes


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


def create_phrase_boxes(bboxes):
    """Create predicate boxes given the subj. and obj. boxes."""
    return np.array([
        [[
            min(bbox[0][0], bbox[1][0]),
            max(bbox[0][1], bbox[1][1]),
            min(bbox[0][2], bbox[1][2]),
            max(bbox[0][3], bbox[1][3])
        ]]
        for bbox in bboxes  # (N, 2, 4)
    ])


def relationship_recall(chkpnts, det_labels, det_bboxes, gt_labels,
                        gt_bboxes, macro_recall=False, phrase_recall=False):
    """
    Evaluate relationship recall, with top n_re predictions per image.

    Inputs:
        - chkpnts: array, thresholds of predictions to keep
        - det_labels: array (Ndet, 3) of detected labels,
            where Ndet is the number of predictions in this image and
            each row: subj_tag, pred_tag, obj_tag]
        - det_bboxes: array (Ndet, 2, 4) of detected boxes,
            where Ndet is the number of predictions in this image and
            each 2x4 array: [
                [y_min_subj, y_max_subj, x_min_subj, x_max_subj]
                [y_min_obj, y_max_obj, x_min_obj, x_max_obj]
            ]
        - gt_labels: array (N, 3) of ground-truth labels,
            where N is the number of ground-truth in this image and
            each row: subj_tag, pred_tag, obj_tag]
        - gt_bboxes: array (N, 2, 4) of ground-truth boxes,
            where N is the number of ground-truth in this image and
            each 2x4 array: [
                [y_min_subj, y_max_subj, x_min_subj, x_max_subj]
                [y_min_obj, y_max_obj, x_min_obj, x_max_obj]
            ]
        - macro_recall: bool, whether to evaluate macro recall
        - phrase_recall: bool, whether to evaluate phrase recall
    Returns:
        - detected positives per top-N threshold
    """
    if phrase_recall:
        det_bboxes = create_phrase_boxes(det_bboxes)
        gt_bboxes = create_phrase_boxes(gt_bboxes)
    relationships_found = np.zeros(chkpnts.shape).astype(np.float)

    # Check only detections that match any of the ground-truth
    possible_matches = (det_labels[..., None] == gt_labels.T[None, ...]).all(1)
    check_inds = possible_matches.any(1)
    for ind, bbox in zip(np.where(check_inds)[0], det_bboxes[check_inds]):
        overlaps = np.array([
            compute_overlap(bbox, gt_box) if match else 0
            for gt_box, match in zip(gt_bboxes, possible_matches[ind])
        ])
        if macro_recall:
            overlaps = np.where(overlaps >= 0.5)[0]
            possible_matches[:, overlaps] = False
            relationships_found[chkpnts > ind] += len(overlaps)
        elif (overlaps >= 0.5).any():  # micro-recall
            possible_matches[:, np.argmax(overlaps)] = False
            relationships_found[chkpnts > ind] += 1
    return relationships_found  # (R@20, R@50, R@100)
