# -*- coding: utf-8 -*-
"""Functions to extract spatial features for objects/relationships."""

import torch


class SpatialFeatureExtractor:
    """Extract spatial features for SGGen-VRD."""

    def __init__(self):
        """Initialize extractor."""
        self.methods = {
            'ben_younes_2019': self._ben_younes_features,
            'gkanatsios_2019b': self._gkanatsiosb_features,
            'gkanatsios_2019b_3d': self._3d_features,
            'hung_2019': self._hung_features,
            'li_2017': self._li_features,
            'shin_2018': self._shin_features,
            'yu_2017': self._yu_features,
            'zhang_2019_reldn': self._zhang_reldn_features,
            'zhang_2017_vtranse': self._zhang_vtranse_features,
            'collell_2018': self. _collell_features
        }

    @staticmethod
    def get_binary_masks(boxes, img_height, img_width, mask_size=32):
        """Create binary masks that are non-zero inside boxes."""
        # TODO: some masks are returned all-zeros (16x16) e.g. '8128429591_3447bbfbf9_b.jpg'
        masks = torch.zeros((len(boxes), 1, mask_size, mask_size))
        h_ratio = float(mask_size) / img_height  # height ratio
        w_ratio = float(mask_size) / img_width  # width ratio
        y_min = torch.clamp(torch.floor(boxes[:, 1] * h_ratio), min=0).int()
        y_max = torch.clamp(
            torch.ceil(boxes[:, 3] * h_ratio), max=mask_size - 1
        ).int() + 1
        x_min = torch.clamp(torch.floor(boxes[:, 0] * w_ratio), min=0).int()
        x_max = torch.clamp(
            torch.ceil(boxes[:, 2] * w_ratio), max=mask_size - 1
        ).int() + 1
        for ind, _ in enumerate(masks):
            masks[ind, 0, y_min[ind]:y_max[ind], x_min[ind]:x_max[ind]] = 1.0
        return masks

    def get_features(self, subj_boxes, obj_boxes, height, width,
                     method, depth=None):
        """Get features of a given method."""
        return self.methods[method](subj_boxes, obj_boxes,
                                    height, width, depth)

    @staticmethod
    def create_pred_boxes(subj_boxes, obj_boxes):
        """Minimum rectangle that encompasses subject and object."""
        dims = subj_boxes.shape[-1]  # in case of 3d features
        return torch.cat((
            torch.min(subj_boxes[:, :dims//2], obj_boxes[:, :dims//2]),
            torch.max(subj_boxes[:, dims//2:], obj_boxes[:, dims//2:])
        ), dim=1)

    def _get_double_deltas(self, boxes_1, boxes_2):
        """
        Create double deltas.

        [(x_subj - x_obj) / w_subj, (y_subj - y_obj) / h_subj]
        + [log(w_subj / w_obj), log(h_subj / h_obj)]
        + [(x_obj - x_subj) / w_obj, (y_obj - y_subj) / h_obj]
        """
        boxes_1 = self._transform_to_centered(boxes_1)
        boxes_2 = self._transform_to_centered(boxes_2)
        if boxes_1.shape[-1] == 6:  # 3d box
            return torch.stack((
                (boxes_1[:, 0] - boxes_2[:, 0]) / boxes_1[:, 3],
                (boxes_1[:, 1] - boxes_2[:, 1]) / boxes_1[:, 4],
                (boxes_1[:, 2] - boxes_2[:, 2]) / boxes_1[:, 5],
                torch.log(boxes_1[:, 3] / boxes_2[:, 3]),
                torch.log(boxes_1[:, 4] * boxes_2[:, 4]),
                torch.log(boxes_1[:, 5] * boxes_2[:, 5]),
                (boxes_2[:, 0] - boxes_1[:, 0]) / boxes_2[:, 3],
                (boxes_2[:, 1] - boxes_1[:, 1]) / boxes_2[:, 4],
                (boxes_2[:, 2] - boxes_1[:, 2]) / boxes_2[:, 5],
            ), dim=1)
        return torch.stack((
            (boxes_1[:, 0] - boxes_2[:, 0]) / boxes_1[:, 2],
            (boxes_1[:, 1] - boxes_2[:, 1]) / boxes_1[:, 3],
            torch.log(boxes_1[:, 2] / boxes_2[:, 2]),
            torch.log(boxes_1[:, 3] / boxes_2[:, 3]),
            (boxes_2[:, 0] - boxes_1[:, 0]) / boxes_2[:, 2],
            (boxes_2[:, 1] - boxes_1[:, 1]) / boxes_2[:, 3],
        ), dim=1)

    def _get_deltas(self, boxes_1, boxes_2):
        """
        Create double deltas.

        [(x_subj - x_obj) / w_subj, (y_subj - y_obj) / h_subj]
        + [log(w_subj / w_obj), log(h_subj * h_obj)]
        """
        boxes_1 = self._transform_to_centered(boxes_1)
        boxes_2 = self._transform_to_centered(boxes_2)
        if boxes_1.shape[-1] == 6:  # 3d box
            return torch.stack((
                (boxes_1[:, 0] - boxes_2[:, 0]) / boxes_1[:, 3],
                (boxes_1[:, 1] - boxes_2[:, 1]) / boxes_1[:, 4],
                (boxes_1[:, 2] - boxes_2[:, 2]) / boxes_1[:, 5],
                torch.log(boxes_1[:, 3] / boxes_2[:, 3]),
                torch.log(boxes_1[:, 4] * boxes_2[:, 4]),
                torch.log(boxes_1[:, 5] * boxes_2[:, 5])
            ), dim=1)
        return torch.stack((
            (boxes_1[:, 0] - boxes_2[:, 0]) / boxes_1[:, 2],
            (boxes_1[:, 1] - boxes_2[:, 1]) / boxes_1[:, 3],
            torch.log(boxes_1[:, 2] / boxes_2[:, 2]),
            torch.log(boxes_1[:, 3] * boxes_2[:, 3])
        ), dim=1)

    def _get_rel_feat(self, subj_boxes, obj_boxes, height, width):
        """Merge to [(xs-xo)/wo, (ys-yo)/ho, ws/wo, hs/ho, as/ao]."""
        subj_boxes = self._normalize_coords(subj_boxes, height, width)
        obj_boxes = self._normalize_coords(obj_boxes, height, width)
        rel_boxes = subj_boxes.clone()
        rel_boxes[:, 2:] /= obj_boxes[:, 2:]  # ws/wo, hs/ho
        rel_boxes[:, :2] -= obj_boxes[:, :2]  # xs-xo), (ys-yo)
        rel_boxes[:, :2] /= obj_boxes[:, 2:]  # xs-xo)/wo, (ys-yo)/ho
        return torch.cat(
            (rel_boxes, (rel_boxes[:, 2] * rel_boxes[:, 3]).unsqueeze(1)),
            dim=1
        )

    @staticmethod
    def _normalize_coords(box, img_height, img_width, img_depth=None):
        """Normalize with image dimensions."""
        box = box.clone()
        if img_depth is not None:
            box[:, (0, 3)] /= img_width
            box[:, (1, 4)] /= img_height
            box[:, (2, 5)] /= img_depth
            return box
        box[:, (0, 2)] /= img_width
        box[:, (1, 3)] /= img_height
        return box

    @staticmethod
    def _transform_to_centered(box):
        """Transform [x1, y1, x2, y2] to [xc, yc, w, h]."""
        if box.shape[-1] == 6:  # 3d box
            return torch.stack((
                (box[:, 3] + box[:, 0]) / 2,
                (box[:, 4] + box[:, 1]) / 2,
                (box[:, 4] + box[:, 2]) / 2,
                box[:, 3] - box[:, 0] + 1,
                box[:, 4] - box[:, 1] + 1,
                box[:, 5] - box[:, 2] + 1
            ), dim=1)
        return torch.stack((
            (box[:, 2] + box[:, 0]) / 2,
            (box[:, 3] + box[:, 1]) / 2,
            box[:, 2] - box[:, 0] + 1,
            box[:, 3] - box[:, 1] + 1
        ), dim=1)

    def _ben_younes_features(self, subj_boxes, obj_boxes,
                             height, width, depth=None):
        """
        Directly use normalized coords.

        [xs_1/W, ys_1/H, xs_2/W, ys_2/H]
        + [xo_1/W, yo_1/H, xo_2/W, yo_2/H]

        See "BLOCK: Bilinear Superdiagonal Fusion for Visual Question
        Answering and Visual Relationship Detection",
        Ben-Younes et al., 2019
        """
        return torch.cat((
            self._normalize_coords(subj_boxes, height, width),
            self._normalize_coords(obj_boxes, height, width)
        ), dim=1)

    def _hung_features(self, subj_boxes, obj_boxes, height, width, depth=None):
        """
        Use normalized coords and log-deltas.

        [xs_1/W, ys_1/H, xs_2/W, ys_2/H, as/A]
        + [xo_1/W, yo_1/H, xo_2/W, yo_2/H, ao/A]
        + [(xs-xo)/wo, (ys-yo)/ho, log(ws/wo), log(hs/ho)]
        + [(xo-xs)/ws, (yo-ys)/hs, log(wo/ws), log(ho/hs)]
        + [Au/A]
        This 19-d vector is fed into a two-layer MLP with intermediate
        layer dimenension of 32 and output dimension of 16.

        See "Union Visual Translation Embedding for Visual
        Relationship Detection and Scene Graph
        Generation"
        Hung et al., 2019
        """
        li_feat = self._li_features(subj_boxes, obj_boxes, height, width)
        pred_boxes = self.create_pred_boxes(subj_boxes, obj_boxes)
        pred_areas = (
            (pred_boxes[:, 3] - pred_boxes[:, 1])
            * (pred_boxes[:, 2] - pred_boxes[:, 0])
        )
        return torch.cat((
            li_feat[:, :4],
            (
                (li_feat[:, 3] - li_feat[:, 1])
                * (li_feat[:, 2] - li_feat[:, 0])
            )[:, None],
            li_feat[:, 4:8],
            (
                (li_feat[:, 7] - li_feat[:, 5])
                * (li_feat[:, 6] - li_feat[:, 4])
            )[:, None],
            li_feat[:, 8:],
            pred_areas[:, None] / (height * width)
        ), dim=1)

    def _li_features(self, subj_boxes, obj_boxes, height, width, depth=None):
        """
        Use normalized coords and log-deltas.

        [xs_1/W, ys_1/H, xs_2/W, ys_2/H]
        + [xo_1/W, yo_1/H, xo_2/W, yo_2/H]
        + [(xs-xo)/wo, (ys-yo)/ho, log(ws/wo), log(hs/ho)]
        + [(xo-xs)/ws, (yo-ys)/hs, log(wo/ws), log(ho/hs)]
        This 16-d vector is concatenated to visual features.

        See "Visual Relationship Detection Using Joint
        Visual-Semantic Embedding"
        Li et al., 2017
        """
        feats = torch.cat((
            self._get_rel_feat(subj_boxes, obj_boxes, height, width)[:, :-1],
            self._get_rel_feat(obj_boxes, subj_boxes, height, width)[:, :-1]
        ), dim=1)
        feats[:, (2, 3, 6, 7)] = torch.log(feats[:, (2, 3, 6, 7)])
        return torch.cat((
            self._ben_younes_features(subj_boxes, obj_boxes, height, width),
            feats
        ), dim=1)

    def _gkanatsiosb_features(self, subj_boxes, obj_boxes,
                              height, width, depth=None):
        """
        Use a fusion of different literature features.

        See "Attention-Translation-Relation Network for Scalable
        Scene Graph Generation"
        Gkanatsios et al., 2019
        """
        pred_boxes = self.create_pred_boxes(subj_boxes, obj_boxes)
        w_subj = subj_boxes[:, 2] - subj_boxes[:, 0]
        h_subj = subj_boxes[:, 3] - subj_boxes[:, 1]
        w_pred = pred_boxes[:, 2] - pred_boxes[:, 0]
        h_pred = pred_boxes[:, 3] - pred_boxes[:, 1]
        w_obj = obj_boxes[:, 2] - obj_boxes[:, 0]
        h_obj = obj_boxes[:, 3] - obj_boxes[:, 1]
        return torch.cat((
            self._get_double_deltas(subj_boxes, obj_boxes),
            self._get_double_deltas(subj_boxes, pred_boxes),
            self._get_double_deltas(obj_boxes, pred_boxes),
            self._normalize_coords(subj_boxes, height, width)[:, (1, 3, 0, 2)],
            self._normalize_coords(obj_boxes, height, width)[:, (1, 3, 0, 2)],
            self._normalize_coords(pred_boxes, height, width)[:, (1, 3, 0, 2)],
            torch.stack((
                w_subj * h_subj / (height * width),
                w_obj * h_obj / (height * width),
                w_pred / w_subj, h_pred / h_subj,
                w_pred / w_obj, h_pred / h_obj,
                w_obj / w_subj, h_obj / h_subj
            ), dim=1)
        ), dim=1)

    def _3d_features(self, subj_boxes, obj_boxes, height, width, depth):
        """
        Spatial features for 3d-box represened objects
        x1, y1, z1, x2, y2, z2
        """
        pred_boxes = self.create_pred_boxes(subj_boxes, obj_boxes)
        w_subj = subj_boxes[:, 3] - subj_boxes[:, 0]
        h_subj = subj_boxes[:, 4] - subj_boxes[:, 1]
        d_subj = subj_boxes[:, 5] - subj_boxes[:, 2]
        w_pred = pred_boxes[:, 3] - pred_boxes[:, 0]
        h_pred = pred_boxes[:, 4] - pred_boxes[:, 1]
        d_pred = pred_boxes[:, 5] - pred_boxes[:, 2]
        w_obj = obj_boxes[:, 3] - obj_boxes[:, 0]
        h_obj = obj_boxes[:, 4] - obj_boxes[:, 1]
        d_obj = obj_boxes[:, 5] - obj_boxes[:, 2]
        return torch.cat((
            self._get_double_deltas(subj_boxes, obj_boxes),
            self._get_double_deltas(subj_boxes, pred_boxes),
            self._get_double_deltas(obj_boxes, pred_boxes),
            self._normalize_coords(subj_boxes, height, width, depth),
            self._normalize_coords(obj_boxes, height, width, depth),
            self._normalize_coords(pred_boxes, height, width, depth),
            torch.stack((
                w_subj * h_subj / (height * width),
                w_obj * h_obj / (height * width),
                d_subj * h_subj / (height * depth),
                d_obj * h_obj / (height * depth),
                d_subj * w_subj / (width * depth),
                d_obj * w_obj / (width * depth),
                w_pred / w_subj, h_pred / h_subj, d_pred / d_subj,
                w_pred / w_obj, h_pred / h_obj, d_pred / d_obj,
                w_obj / w_subj, h_obj / h_subj, d_obj / d_subj
            ), dim=1)
        ), dim=1)

    def _shin_features(self, subj_boxes, obj_boxes, height, width, depth=None):
        """
        Use centered normalized coords.

        [xs_c/W, ys_c/H, ws/W, hs/H] + [xo_c/W, yo_c/H, wo/W, ho/H]

        See "Deep Image Understanding Using Multilayered Contexts"
        Shin et al., 2018
        """
        return torch.cat((
            self._normalize_coords(
                self._transform_to_centered(subj_boxes), height, width),
            self._normalize_coords(
                self._transform_to_centered(obj_boxes), height, width)
        ), dim=1)

    def _yu_features(self, subj_boxes, obj_boxes, height, width, depth=None):
        """
        Use centered normalized coords and areas.

        [xs_c/W, ys_c/H, ws/W, hs/H, as/A]
        + [xo_c/W, yo_c/H, wo/W, ho/H, ao/A]

        See "Visual Relationship Detection with Internal and External
        Linguistic Knowledge Distillation"
        Yu et al., 2017
        """
        boxes = self._shin_features(subj_boxes, obj_boxes, height, width)
        return torch.cat((
            boxes[:, :4], (boxes[:, 2] * boxes[:, 3]).unsqueeze(1),
            boxes[:, 4:], (boxes[:, 6] * boxes[:, 7]).unsqueeze(1)
        ), dim=1)

    def _zhang_reldn_features(self, subj_boxes, obj_boxes,
                              height, width, depth=None):
        """
        Use normalized coords and log-deltas.

        delta(s, o) + delta(s, p) + delta(p, o)
        + [xs_1/W, ys_1/H, xs_2/W, ys_2/H, as/A]
        + [xo_1/W, yo_1/H, xo_2/W, yo_2/H, ao/A]
        This 22-d vector is fed into a three-layer MLP with intermediate
        layer dimenensions of 64 and output dimension of num_classes.

        See "Graphical Contrastive Losses for Scene Graph Generation"
        Zhang et al., 2019
        """
        pred_boxes = self.create_pred_boxes(subj_boxes, obj_boxes)
        return torch.cat((
            self._get_deltas(subj_boxes, obj_boxes),
            self._get_deltas(subj_boxes, pred_boxes),
            self._get_deltas(pred_boxes, obj_boxes),
            self._hung_features(subj_boxes, obj_boxes, height, width)[:, :10]
        ), dim=1)

    def _zhang_vtranse_features(self, subj_boxes, obj_boxes,
                                height, width, depth=None):
        """
        Use normalized coords and log-deltas.

        [(xs-xo)/wo, (ys-yo)/ho, log(ws/wo), log(hs/ho)]
        + [(xo-xs)/ws, (yo-ys)/hs, log(wo/ws), log(ho/hs)]

        See "Visual Translation Embedding Network for
        Visual Relation Detection"
        Zhang et al., 2017
        """
        return torch.cat((
            self._get_double_deltas(subj_boxes, obj_boxes)[:, :4],
            self._get_double_deltas(obj_boxes, subj_boxes)[:, :4]
        ), dim=1)

    def _collell_features(self, subj_boxes, obj_boxes, height, width, depth=None):
        """
        Use centered normalized coords.

        [xs_c/W, ys_c/H, ws/(2* W), hs/(2* H)],
        [xo_c/W, yo_c/H, wo/(2* W), ho/(2* H)]

        See "Acquiring Common Sense Spatial
             Knowledge through Implicit Spatial Templates"
        Collell et al., 2018
        """
        subj_nrm_boxes = self._normalize_coords(
                self._transform_to_centered(
                    subj_boxes.type(torch.float)), height, width)
        subj_nrm_boxes[:, 2:] /= 2
        obj_nrm_boxes = self._normalize_coords(
                self._transform_to_centered(
                    obj_boxes.type(torch.float)), height, width)
        obj_nrm_boxes[:, 2:] /= 2
        return subj_nrm_boxes, obj_nrm_boxes
