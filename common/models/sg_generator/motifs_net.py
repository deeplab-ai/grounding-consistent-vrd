# -*- coding: utf-8 -*-
"""Neural Motifs Network by Zellers et al., 2018."""

import torch
from torch import nn
from torch.nn import functional as F

from .base_sg_generator import BaseSGGenerator


class MotifsNet(BaseSGGenerator):
    """Extends PyTorch nn.Module."""

    def __init__(self, config):
        """Initialize layers."""
        super().__init__(
            config,
            {'base_features', 'object_1hots', 'object_masks', 'pool_features'}
        )

        # Object context
        self.obj_projector = nn.Sequential(nn.Linear(1024, 512), nn.ReLU())
        self.cls_projector = nn.Sequential(
            nn.Linear(self.num_obj_classes, 100), nn.ReLU()
        )
        self.obj_cntxt = AlternatingHighwayLSTM(512 + 100, 256)
        # Rel context
        self.cls_projector2 = nn.Sequential(
            nn.Linear(self.num_obj_classes, 100), nn.ReLU()
        )
        self.rel_cntxt = AlternatingHighwayLSTM(256 + 100, 256)
        # Fusion of features
        self.fc_subject = nn.Linear(256, 512)
        self.fc_predicate = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(),
            nn.Linear(256, 512)
        )
        self.fc_object = nn.Linear(256, 512)
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 256, 8), nn.ReLU()
        )
        self.mask_net2 = nn.Linear(256, 512)
        # Classifiers
        self.fc_classifier = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, self.num_rel_classes)
        )
        self.fc_classifier_bin = nn.Sequential(
            nn.Linear(512, 128), nn.ReLU(),
            nn.Linear(128, 2)
        )

    def contextualize(self, objects, base_features):
        """Refine object features using structured motifs."""
        obj_feats = objects['pool_features']
        obj_1hots = objects['1hots']
        # Object context refinement (before obj. classification)
        obj_feats = torch.cat(
            (self.obj_projector(obj_feats), self.cls_projector(obj_1hots)),
            dim=1
        )  # total dim 512 + 100
        obj_feats = self.obj_cntxt(obj_feats)  # dim 256
        # Refinement after obj. classification
        obj_feats = torch.cat(
            (obj_feats, self.cls_projector2(obj_1hots)),
            dim=1
        )  # total dim 256 + 100
        obj_feats = self.rel_cntxt(obj_feats)  # dim 256
        objects['pool_features'] = obj_feats
        return objects

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
            objects['pool_features'][pairs[:, 0]],
            self.get_pred_pooled_features(
                base_features,
                objects['boxes'][pairs[:, 0]], objects['boxes'][pairs[:, 1]]
            ),
            objects['pool_features'][pairs[:, 1]],
            # self.get_pred_probabilities(
            #     objects['ids'][pairs[:, 0]], objects['ids'][pairs[:, 1]]
            # ),
            objects['masks'][pairs[:, 0]],
            objects['masks'][pairs[:, 1]]
        )

    def _forward(self, subj_feats, pred_feats, obj_feats, subj_masks,
                 obj_masks):
        """Forward pass, returns output scores."""
        # Predicate features
        masks = torch.cat((subj_masks, obj_masks), dim=1)
        spat_feats = self.mask_net2(self.mask_net(masks).squeeze(2).squeeze(2))
        pred_feats = self.fc_predicate(pred_feats)
        pred_feats = pred_feats + spat_feats
        # S-O features
        subj_feats = self.fc_subject(subj_feats)
        obj_feats = self.fc_object(obj_feats)
        # Fusion with Hadamard product
        fused_feats = subj_feats * pred_feats * obj_feats
        # Classification
        scores = self.fc_classifier(fused_feats)
        scores_bin = self.fc_classifier_bin(fused_feats)
        if self.mode == 'test':
            scores = self.softmax(scores)
            scores_bin = self.softmax(scores_bin)
        return scores, scores_bin


class AlternatingHighwayLSTM(nn.Module):
    """Alternating highway LSTM."""

    def __init__(self, input_size, output_size, layers=2):
        """Initialize layer."""
        super().__init__()
        self.cells = nn.ModuleList([
            nn.LSTMCell(input_size, output_size) if i == 0
            else nn.LSTMCell(output_size, output_size)
            for i in range(layers)
        ])
        self.transform_gates = nn.ModuleList([
            nn.Linear(input_size + output_size, output_size) if i == 0
            else nn.Linear(2 * output_size, output_size)
            for i in range(layers)
        ])
        self.highways = nn.ModuleList([
            nn.Linear(input_size, output_size, bias=False) if i == 0
            else nn.Linear(output_size, output_size, bias=False)
            for i in range(layers)
        ])

    def forward(self, sequence):
        """Forward pass for a sequence (N_data, input_size)."""
        # out_sequence = sequence[::-1]  # convenient trick
        out_sequence = torch.flip(sequence, (0,))
        for cnt in range(len(self.cells)):
            cell = self.cells[cnt]
            gate = self.transform_gates[cnt]
            highway = self.highways[cnt]
            # inp_sequence = out_sequence[::-1]  # alternating directions
            inp_sequence = torch.flip(out_sequence, (0,))
            h_state = torch.zeros(1, 256).to(sequence.device)
            c_state = torch.zeros(1, 256).to(sequence.device)
            out_sequence = []
            for item in inp_sequence:
                item = item.unsqueeze(0)
                h_out, c_state = cell(item, (h_state, c_state))
                wght = F.sigmoid(gate(torch.cat((h_state, item), dim=1)))
                h_state = wght * h_out + (1 - wght) * highway(item)
                out_sequence.append(h_state)
            out_sequence = torch.cat(out_sequence)
        if len(self.cells) % 2:
            out_sequence = out_sequence[::-1]
        return out_sequence
