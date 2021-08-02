# -*- coding: utf-8 -*-
"""
A simple referring relationships implementation for grounding
"""

import torch
from torch import nn
import torch.nn.functional as F
from torchvision.ops import MultiScaleRoIAlign

from .base_grounder import BaseGRNDGenerator


class ParsingNet(BaseGRNDGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config, features={}):
        """Initialize layers."""
        super().__init__(config, {'base_features', 'pool_features',
                                  'object_masks'}, mask_size=15)
        self._set_norm_centers()
        self.subj_regressor = BBoxReg()
        self.obj_regressor = BBoxReg()
        self.img_roi_pool = MultiScaleRoIAlign(
                            featmap_names=['0', '1', '2', '3'],
                            output_size=self._mask_size,
                            sampling_ratio=2)
        self.vis_pooling = VisAttentionalPooling(256)
        self.fc_subj = nn.Sequential(nn.Linear(300, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU(),
                                     nn.Linear(256, 256), nn.ReLU())
        self.fc_obj = nn.Sequential(nn.Linear(300, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU(),
                                    nn.Linear(256, 256), nn.ReLU())
        self.feats_conv = nn.Sequential(
                            nn.Conv2d(256, 256, kernel_size=(1, 1)),
                            nn.ReLU()
        )
        self.conv_subj = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(256 + 38, 128, kernel_size=(1, 1)),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=(1, 1)), nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=(1, 1)), nn.BatchNorm2d(1),
                nn.ReLU(), nn.Tanh())
             for _ in range(self.num_rel_classes)]
        )
        self.conv_obj = nn.ModuleList(
            [nn.Sequential(
                nn.Conv2d(256 + 38, 128, kernel_size=(1, 1)),
                nn.BatchNorm2d(128), nn.ReLU(),
                nn.Conv2d(128, 64, kernel_size=(1, 1)), nn.BatchNorm2d(64),
                nn.ReLU(),
                nn.Conv2d(64, 1, kernel_size=(1, 1)), nn.BatchNorm2d(1),
                nn.ReLU(), nn.Tanh())
             for _ in range(self.num_rel_classes)]
        )

    def net_forward(self, base_features, objects, pairs, predicate_ids):
        """Forward pass, override."""
        # Get image features
        # [x1, y1, x2, y2] format
        scales = self._box_scales
        img_box = torch.FloatTensor(
            [[0, 0, self._img_shape[1] / scales[1],
              self._img_shape[0] / scales[0]]]).to(self._device)
        rois = self._rescale_boxes(img_box, self._box_scales)
        img_feats = self.img_roi_pool(base_features, [rois], [self._img_shape])

        # Get normalized box features
        subj_boxes = objects['boxes'][pairs[:, 0]]
        obj_boxes = objects['boxes'][pairs[:, 1]]
        (
            nrm_subj_boxes, nrm_obj_boxes
        ) = self.spatial_extractor.get_features(
                subj_boxes, obj_boxes,
                self._image_info[1], self._image_info[0], 'collell_2018'
            )
        return self._forward(
            img_feats,
            self.get_obj_embeddings(objects['ids'][pairs[:, 0]]),
            self.get_pred_embeddings(predicate_ids),
            self.get_obj_embeddings(objects['ids'][pairs[:, 1]]),
            nrm_subj_boxes,
            nrm_obj_boxes,
            predicate_ids,
            subj_boxes,
            obj_boxes
        )

    def _forward(self, img_feats, subj_embs, pred_embs, obj_embs,
                 nrm_subj_boxes, nrm_obj_boxes, predicate_ids,
                 subj_boxes, obj_boxes):
        """Forward pass, returns output scores."""
        # Predict box width, height
        nrm_subj_box_pred = self.subj_regressor(
            subj_embs, pred_embs, obj_embs, nrm_obj_boxes)
        nrm_obj_box_pred = self.obj_regressor(
            subj_embs, pred_embs, obj_embs, nrm_subj_boxes)

        # Compute b-box language-guided visual features
        img_feats = self.feats_conv(img_feats)
        subj_embs = self.fc_subj(subj_embs)
        obj_embs = self.fc_subj(obj_embs)
        subj_att_feats, subj_att = self.vis_pooling(img_feats, subj_embs)
        obj_att_feats, obj_att = self.vis_pooling(img_feats, obj_embs)
        subj_roi_feats = self.spatial_parse(
            subj_att_feats.data, nrm_subj_box_pred)
        obj_roi_feats = self.spatial_parse(
            obj_att_feats.data, nrm_obj_box_pred)

        obj_spat_feats = self.get_spatial_features(
            subj_boxes, nrm_obj_box_pred)
        subj_spat_feats = self.get_spatial_features(
            obj_boxes, nrm_subj_box_pred, invert=True)

        subj_featmap = torch.cat((subj_roi_feats, subj_spat_feats), dim=1)
        obj_featmap = torch.cat((obj_roi_feats, obj_spat_feats), dim=1)

        subj_hmap = self.head_forward(
            subj_featmap, self.conv_subj, predicate_ids).squeeze(1)
        obj_hmap = self.head_forward(
            obj_featmap, self.conv_obj, predicate_ids).squeeze(1)

        if self.mode == 'test':
            return (
                subj_hmap * subj_att,
                obj_hmap * obj_att,
                subj_att, obj_att,
                nrm_subj_box_pred, nrm_obj_box_pred
            )
        return (
            subj_hmap, obj_hmap, subj_att, obj_att,
            nrm_subj_box_pred, nrm_obj_box_pred
        )

    def spatial_parse(self, att_feats, nrm_box):
        """Convolve predicted binary bbox mask with features"""
        # width or height cannot be >= 0.5
        nrm_box_clmp = torch.clamp(nrm_box.data, 0.0, 0.49)
        K = torch.zeros((nrm_box.shape[0], self._mask_size,
                         self._mask_size)).to(self._device)
        for i in range(K.shape[0]):
            W = torch.round(self._mask_size * nrm_box_clmp[i, 0].data
                            ).type(torch.long)
            x_0 = self._mask_size // 2 - W
            x_1 = self._mask_size // 2 + W + 1
            H = torch.round(self._mask_size * nrm_box_clmp[i, 1].data
                            ).type(torch.long)
            y_0 = self._mask_size // 2 - H
            y_1 = self._mask_size // 2 + H + 1
            K[i, y_0:y_1, x_0:x_1] = 1
        shape = att_feats.shape
        att_feats = att_feats.view(1, -1, shape[2], shape[3])
        K = K.unsqueeze(1).repeat_interleave(shape[1], dim=0)
        agreement_map = F.conv2d(att_feats, K, groups=att_feats.shape[1],
                                 padding=shape[2]//2)
        # Normalize
        agreement_map_nrm = agreement_map / K.sum(3).sum(2).view(
            1, 1, agreement_map.shape[1], 1, 1)
        return agreement_map_nrm.view(-1, shape[1], shape[2], shape[3])

    def head_forward(self, features, modules, pred_ids):
        outputs = []
        for i, feat in enumerate(features):
            outputs.append(modules[pred_ids[i]](feat.unsqueeze(0)))
        outputs = torch.cat(outputs, dim=0)
        return outputs

    def _set_norm_centers(self):
        # Create normalized spatial centers
        flat_concat = lambda xy: torch.stack((
            xy[0].t().flatten(0), xy[1].t().flatten(0),
            xy[0].t().flatten(0), xy[1].t().flatten(0)
        )).t()
        xc = torch.linspace(
            0, 1, self._mask_size).to(self._device)
        yc = torch.linspace(
            0, 1, self._mask_size).to(self._device)
        xc_yc = flat_concat(torch.meshgrid((xc, yc)))
        self.xc_yc_norm = xc_yc

    def get_spatial_features(self, boxes, nrm_box_dims, invert=False):
        # Create spatial features
        W, H = self._image_info
        xc_yc = self.xc_yc_norm.clone()
        xc_yc[:, (0, 2)] *= W
        xc_yc[:, (1, 3)] *= H
        box_dims = nrm_box_dims.data.clone()
        box_dims = torch.cat((box_dims, box_dims), dim=1)
        box_dims[:, 0] *= -W
        box_dims[:, 1] *= -H
        box_dims[:, 2] *= W
        box_dims[:, 3] *= H
        slide_boxes = xc_yc.repeat(len(boxes), 1, 1) + box_dims.unsqueeze(1)

        if not invert:
            spatial_features =\
                self.spatial_extractor.get_features(
                    boxes.unsqueeze(1).expand(
                        -1, self._mask_size**2, -1).reshape(-1, 4),
                    slide_boxes.view(-1, 4),
                    H, W, 'gkanatsios_2019b')
        else:
            spatial_features =\
                self.spatial_extractor.get_features(
                    slide_boxes.view(-1, 4),
                    boxes.unsqueeze(1).expand(
                        -1, self._mask_size**2, -1).reshape(-1, 4),
                    H, W, 'gkanatsios_2019b')
        spatial_features = spatial_features.view(
            len(boxes), self._mask_size, self._mask_size, -1
        ).permute(0, 3, 1, 2)
        return spatial_features


class BBoxReg(nn.Module):
    """Perform b-box regression on width and height"""

    def __init__(self):
        """Initialize layers."""
        super().__init__()
        self.fc_subject = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_predicate = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_object = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        # NOTE: add sigmoid here
        self.fc_regressor = nn.Sequential(nn.Linear(770, 256), nn.ReLU(),
                                          nn.Linear(256, 128), nn.ReLU(),
                                          nn.Linear(128, 2))

    def forward(self, subj_embs, pred_embs, obj_embs, nrm_boxes):
        embeddings = torch.cat((
            self.fc_subject(subj_embs), self.fc_predicate(pred_embs),
            self.fc_object(obj_embs), nrm_boxes[:, 2:]
        ), dim=1)
        y = self.fc_regressor(embeddings)
        return torch.sigmoid(y) / 2


class VisAttentionalPooling(nn.Module):
    """Visual attentional pooling implementation."""

    def __init__(self, input_dim):
        """Initialize layers."""
        super().__init__()
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, feature_map, embeddings):
        """Forward pass."""
        att_map = (feature_map * embeddings.unsqueeze(-1).unsqueeze(-1)).sum(1)
        att_map = self.tanh(self.relu(att_map))
        att_feats = att_map.unsqueeze(1) * feature_map
        return att_feats, att_map
