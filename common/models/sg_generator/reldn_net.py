# -*- coding: utf-8 -*-
"""Relationship Detection Network by Zhang et al., 2019."""

import torch
from torch import nn

from .base_sg_generator import BaseSGGenerator


class RelDN(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(config, {'base_features', 'pool_features'})

        # Visual features
        self.fc_subject = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                        nn.Linear(512, 256), nn.ReLU())
        self.fc_predicate = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                          nn.Linear(512, 256), nn.ReLU())
        self.fc_object = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(),
                                       nn.Linear(512, 256), nn.ReLU())
        self.pred_classifier = nn.Sequential(
            nn.Linear(768, 512), nn.ReLU(),
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, self.num_rel_classes)
        )
        self.subj_classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                             nn.Linear(128, self.num_rel_classes))
        self.obj_classifier = nn.Sequential(nn.Linear(256, 128), nn.ReLU(),
                                            nn.Linear(128, self.num_rel_classes))

        # Spatial features
        self.delta_net = nn.Sequential(
            nn.Linear(38, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU()
        )
        self.spatial_classifier = nn.Sequential(nn.Linear(64, 64), nn.ReLU(),
                                                nn.Linear(64, self.num_rel_classes))

        self.fc_classifier_bin = nn.Sequential(
            nn.Linear(self.num_rel_classes, 128), nn.ReLU(),
            nn.Linear(128, 2)
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
            self.get_pred_probabilities(
                objects['ids'][pairs[:, 0]], objects['ids'][pairs[:, 1]]
            ),
            self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]],
                method='gkanatsios_2019b'
            )
        )

    def _forward(self, subj_feats, pred_feats, obj_feats, probabilities,
                 spat_feats):
        """Forward pass, returns output scores."""
        # Feature processing and deep scores
        visual_scores = self.visual_forward(subj_feats, pred_feats, obj_feats)
        spat_scores = self.spatial_forward(spat_feats)
        sem_scores = self.semantic_forward(probabilities)

        # Classification
        scores = visual_scores + spat_scores + sem_scores
        scores_bin = self.fc_classifier_bin(scores)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin, visual_scores, spat_scores, sem_scores

    def semantic_forward(self, probs):
        """Forward of semantic net."""
        return torch.log(probs)

    def spatial_forward(self, spat_feats):
        """Forward of spatial net."""
        features = self.delta_net(spat_feats)
        scores = self.spatial_classifier(features)
        return scores

    def visual_forward(self, subj_feats, pred_feats, obj_feats):
        """Forward of visual net."""
        subj_feats = self.fc_subject(subj_feats)
        pred_feats = self.fc_predicate(pred_feats)
        obj_feats = self.fc_object(obj_feats)
        pred_scores = self.pred_classifier(torch.cat((
            subj_feats, pred_feats, obj_feats
        ), dim=1))
        subj_scores = self.subj_classifier(subj_feats)
        obj_scores = self.obj_classifier(obj_feats)
        return subj_scores + pred_scores + obj_scores
