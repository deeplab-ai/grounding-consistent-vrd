# -*- coding: utf-8 -*-
"""Hierarchical Graph Attention Network, by Mi et al., 2020."""

import torch
from torch import nn
from torch.nn import functional as F

from .base_sg_generator import BaseSGGenerator


class HGATNet(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(config, {'base_features', 'pool_features'})

        # Object context
        self.obj_projector = nn.Sequential(
            nn.Linear(1024 + 300, 512), nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.inner_projector1 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128, bias=False)
        )
        self.inner_projector2 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.cntxt_obj_projector = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 256)
        )
        # Rel context
        self.rel_projector = nn.Sequential(
            nn.Linear(1024 + 300 +  256 + 1024 + 300 + 256 + 38, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )
        self.inner_rel_projector1 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128, bias=False)
        )
        self.inner_rel_projector2 = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128)
        )
        self.cntxt_rel_projector = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 256)
        )
        # Classifiers
        self.fc_classifier = nn.Sequential(
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, self.num_rel_classes)
        )
        self.fc_classifier_bin = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def contextualize(self, objects, base_features):
        """Refine object features using structured motifs."""
        obj_feats = self.obj_projector(torch.cat(
            (objects['pool_features'], self.get_obj_embeddings(objects['ids'])),
            dim=1
        ))
        proj1_feats = self.inner_projector1(obj_feats)
        proj2_feats = self.inner_projector2(obj_feats)
        alphas = torch.mm(proj1_feats, proj2_feats.T)
        alphas.fill_diagonal_(0)
        alphas /= (alphas.sum(1).view(-1, 1) + 1e-8)
        # Object context refinement (before obj. classification)
        obj_feats = F.relu(
            obj_feats
            + torch.matmul(alphas, self.cntxt_obj_projector(obj_feats))
        )
        objects['ref_features'] = obj_feats
        return objects

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
            objects['pool_features'][pairs[:, 0]],
            objects['pool_features'][pairs[:, 1]],
            objects['ref_features'][pairs[:, 0]],
            objects['ref_features'][pairs[:, 1]],
            self.get_obj_embeddings(objects['ids'][pairs[:, 0]]),
            self.get_obj_embeddings(objects['ids'][pairs[:, 1]]),
            self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]]
            )
        )

    def _forward(self, subj_feats, obj_feats, ref_subj_feats, ref_obj_feats,
                 subj_emb, obj_emb, spat_feats):
        """Forward pass, returns output scores."""
        # Predicate features
        feats = self.rel_projector(torch.cat(
            (subj_feats, obj_feats, subj_emb, obj_emb, ref_subj_feats,
             ref_obj_feats, spat_feats),
            dim=1
        ))
        proj1_feats = self.inner_rel_projector1(feats)
        proj2_feats = self.inner_rel_projector2(feats)
        alphas = torch.mm(proj1_feats, proj2_feats.T)
        alphas.fill_diagonal_(0)
        alphas /= (alphas.sum(1).view(-1, 1) + 1e-8)
        feats = F.relu(
            feats
            + torch.matmul(alphas, self.cntxt_rel_projector(feats))
        )
        # Classification
        scores = self.fc_classifier(feats)
        scores_bin = self.fc_classifier_bin(feats)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin
