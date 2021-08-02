# -*- coding: utf-8 -*-
"""Union Visual Translation Embeddings net by Hung et al., 2019."""

import torch
from torch import nn
from torch.nn import functional as F

from .base_sg_generator import BaseSGGenerator


class UVTransE(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(config, {'base_features', 'pool_features'})

        # Visual encoder
        self.fc_subject = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_predicate = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.fc_object = nn.Sequential(
            nn.Linear(1024, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU()
        )
        self.spatial_net = nn.Sequential(
            nn.Linear(19, 32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU()
        )
        self.v_encoder = nn.Sequential(nn.Linear(256 + 16, 256), nn.ReLU())
        self.v_classifier = nn.Linear(256, self.num_rel_classes)

        # Language decoder
        self.visual_projector = nn.Sequential(nn.Linear(256, 300), nn.ReLU())
        self.gru = nn.GRU(300, 100, num_layers=1, bidirectional=True)
        self.l_classifier = nn.Sequential(
            nn.Linear(600, 256), nn.ReLU(),
            nn.Linear(256, self.num_rel_classes)
        )
        self.l_bin_classifier = nn.Sequential(
            nn.Linear(600, 256), nn.ReLU(),
            nn.Linear(256, 2)
        )

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
            objects['pool_features'][pairs[:, 0]],
            self.get_pred_pooled_features(
                base_features,
                objects['boxes'][pairs[:, 0]], objects['boxes'][pairs[:, 1]]
            ),
            objects['pool_features'][pairs[:, 1]],
            self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]],
                method='hung_2019'
            ),
            self.get_obj_embeddings(objects['ids'][pairs[:, 0]]),
            self.get_obj_embeddings(objects['ids'][pairs[:, 1]])
        )

    def _forward(self, subj_feats, pred_feats, obj_feats,
                 spat_feats, subj_embs, obj_embs):
        """Forward pass, return output scores."""
        # Visual encoding
        subj_feats = self.fc_subject(subj_feats)
        obj_feats = self.fc_object(obj_feats)
        pred_feats = self.fc_predicate(pred_feats)
        spat_feats = self.spatial_net(spat_feats)
        encoded_feats = self.v_encoder(
            torch.cat((pred_feats - subj_feats - obj_feats, spat_feats), dim=1)
        )
        visual_scores = self.v_classifier(encoded_feats)

        # Language decoding
        triplets = torch.stack(
            (subj_embs, self.visual_projector(encoded_feats), obj_embs), dim=0
        )
        gru_out, _ = self.gru(triplets)
        gru_out = torch.cat(
            (gru_out[0, :, :], gru_out[1, :, :], gru_out[2, :, :]), dim=1
        )
        language_scores = self.l_classifier(gru_out)
        bin_scores = self.l_bin_classifier(gru_out)

        if self.mode == 'test':
            visual_scores = self.softmax(visual_scores)
            language_scores = self.softmax(language_scores)
            bin_scores = self.softmax(bin_scores)

        return (
            0.5 * visual_scores + 0.5 * language_scores,
            bin_scores, visual_scores, language_scores,
            subj_feats, pred_feats, obj_feats
        )
