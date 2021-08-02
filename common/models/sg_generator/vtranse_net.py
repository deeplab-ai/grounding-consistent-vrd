# -*- coding: utf-8 -*-
"""GPS-Net: Graph Property Sensing Network by Lin 2020."""

import torch
from torch import nn
from .base_sg_generator import BaseSGGenerator


class VTransENet(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(config, {'base_features', 'pool_features',
                                  'object_1hots'})  # should be classemes
        input_dim = 1024 + self.num_obj_classes + 4
        self.Ws = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.Wo = nn.Sequential(nn.Linear(input_dim, 128), nn.ReLU())
        self.Wp = nn.Linear(128, self.num_rel_classes)
        self.Wp_bin = nn.Linear(128, 2)

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        spat_feats = self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]],
                method='zhang_2017_vtranse'
        )
        return self._forward(
            objects['pool_features'][pairs[:, 0]],
            objects['pool_features'][pairs[:, 1]],
            objects['1hots'][pairs[:, 0]],
            objects['1hots'][pairs[:, 1]],
            spat_feats[:, :4],
            spat_feats[:, 4:]
        )

    def _forward(self, subj_feats, obj_feats, subj_1hot, obj_1hot,
                 subj_spat_feats, obj_spat_feats):
        """Forward pass, return output scores."""
        subj_feats = self.Ws(
            torch.cat((subj_feats, subj_1hot, subj_spat_feats), dim=1))
        obj_feats = self.Wo(
            torch.cat((obj_feats, obj_1hot, obj_spat_feats), dim=1))
        scores = self.Wp(subj_feats - obj_feats)
        scores_bin = self.Wp_bin(subj_feats - obj_feats)
        if self.mode == 'test':
            return self.softmax(scores), self.softmax(scores_bin)
        return scores, scores_bin
