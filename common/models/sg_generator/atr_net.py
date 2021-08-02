# -*- coding: utf-8 -*-
"""Attention-Translation-Relation Network, Gkanatsios et al., 2019."""

import torch
from torch import nn

from .base_sg_generator import BaseSGGenerator


class ATRNet(BaseSGGenerator):
    """ATRNet/MATransE main."""

    def __init__(self, config, attention='multi_head',
                 use_language=True, use_spatial=True, **kwargs):
        """Initialize model."""
        super().__init__(
            config, {'base_features', 'object_masks', 'pool_features'},
            **kwargs
        )
        self.p_branch = PredicateBranch(
            self.num_rel_classes, attention,
            use_language=use_language, use_spatial=use_spatial)
        self.os_branch = ObjectSubjectBranch(
            self.num_rel_classes, attention,
            use_language=use_language, use_spatial=use_spatial)
        self.fc_fusion = nn.Sequential(
            nn.Linear(2 * self.num_rel_classes, 100), nn.ReLU(),
            nn.Linear(100, self.num_rel_classes)
        )
        self.fc_bin_fusion = nn.Linear(2 * 2, 2)

    def net_forward(self, base_features, objects, pairs):
        """Forward pass, override."""
        return self._forward(
            objects['pool_features'][pairs[:, 0]],
            self.get_roi_features(
                base_features,
                objects['boxes'][pairs[:, 0]], objects['boxes'][pairs[:, 1]]
            ),
            objects['pool_features'][pairs[:, 1]],
            self.get_spatial_features(
                objects['boxes'][pairs[:, 0]],
                objects['boxes'][pairs[:, 1]],
                method='gkanatsios_2019b'
            ),
            objects['masks'][pairs[:, 0]],
            objects['masks'][pairs[:, 1]],
            self.get_obj_embeddings(objects['ids'][pairs[:, 0]]),
            self.get_obj_embeddings(objects['ids'][pairs[:, 1]])
        )

    def _forward(self, subj_feats, pred_feats, obj_feats, deltas,
                 subj_masks, obj_masks, subj_embs, obj_embs):
        """Forward pass."""
        masks = torch.cat((subj_masks, obj_masks), dim=1)
        pred_scores, bin_pred_scores = self.p_branch(
            pred_feats, deltas, masks, subj_embs, obj_embs)
        os_scores, bin_os_scores = self.os_branch(
            subj_feats, obj_feats, subj_embs, obj_embs, masks, deltas)
        scores = self.fc_fusion(torch.cat((pred_scores, os_scores), dim=1))
        bin_scores = self.fc_bin_fusion(torch.cat(
            (bin_pred_scores, bin_os_scores), dim=1))
        if self.mode == 'test':  # scores across pairs are compared in R_70
            scores = self.softmax(scores)
            pred_scores = self.softmax(pred_scores)
            os_scores = self.softmax(os_scores)
            bin_scores = self.softmax(bin_scores)
            bin_pred_scores = self.softmax(bin_pred_scores)
            bin_os_scores = self.softmax(bin_os_scores)
        return (
            scores, bin_scores, pred_scores, os_scores,
            bin_pred_scores, bin_os_scores
        )


class PredicateBranch(nn.Module):
    """
    Predicate Branch.

    pred. features -> CONV -> RELU -> Att. Pool -> CONV1D -> RELU ->
    -> Att. Clsfr -> out
    attention: multihead, singlehead, None
    """

    def __init__(self, num_classes, attention='multi_head',
                 use_language=True, use_spatial=True):
        """Initialize model."""
        super().__init__()
        self._attention_type = attention
        self.conv_1 = nn.Sequential(
            nn.Conv2d(256, 1024, 1), nn.ReLU(),
            nn.Conv2d(1024, 256, 1), nn.ReLU())
        self.conv_1b = nn.Sequential(
            nn.Conv2d(256, 1024, 1), nn.ReLU(),
            nn.Conv2d(1024, 256, 1), nn.ReLU())
        self.attention_layer = AttentionLayer(
            use_language and attention is not None,
            use_spatial and attention is not None)
        self.pooling_weights = AttentionalWeights(
            num_classes, feature_dim=256, attention_type=attention)
        self.binary_pooling_weights = AttentionalWeights(
            2, feature_dim=256, attention_type=attention)
        self.attentional_pooling = AttentionalPoolingLayer()
        self.conv_2 = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU())
        self.conv_2b = nn.Sequential(nn.Conv1d(256, 128, 1), nn.ReLU())
        if attention == 'multi_head':
            self.classifier_weights = AttentionalWeights(
                num_classes, feature_dim=128, attention_type=attention)
            self.bias = nn.Parameter(torch.rand(1, num_classes))
            self.binary_classifier_weights = AttentionalWeights(
                2, feature_dim=128, attention_type=attention)
            self.binary_bias = nn.Parameter(torch.rand(1, 2))
        else:
            self.classifier_weights = nn.Linear(128, num_classes)
            self.binary_classifier_weights = nn.Linear(128, 2)

    def forward(self, pred_feats, deltas, masks, subj_embs, obj_embs):
        """Forward pass."""
        attention = self.attention_layer(subj_embs, obj_embs, deltas, masks)
        conv_pred_feats = self.conv_1(pred_feats)
        bin_conv_pred_feats = self.conv_1b(pred_feats)
        if self._attention_type is not None:
            pred_feats = self.attentional_pooling(
                conv_pred_feats,
                self.pooling_weights(attention)
            )
            bin_pred_feats = self.attentional_pooling(
                bin_conv_pred_feats,
                self.binary_pooling_weights(attention)
            )
        else:
            pred_feats = conv_pred_feats.mean(3).mean(2).unsqueeze(-1)
            bin_pred_feats = bin_conv_pred_feats.mean(3).mean(2).unsqueeze(-1)
        pred_feats = self.conv_2(pred_feats)
        bin_pred_feats = self.conv_2b(bin_pred_feats)
        if self._attention_type == 'multi_head':
            return (
                torch.sum(
                    pred_feats * self.classifier_weights(attention),
                    dim=1)
                + self.bias,
                torch.sum(
                    bin_pred_feats * self.binary_classifier_weights(attention),
                    dim=1)
                + self.binary_bias
            )
        return (
            self.classifier_weights(pred_feats.view(-1, 128)),
            self.binary_classifier_weights(bin_pred_feats.view(-1, 128))
        )


class ObjectSubjectBranch(nn.Module):
    """
    Object-Subject Branch.

    obj. features  -> FC -> RELU -> FC -> RELU -> |
                                                  - -> Att. Clsfr -> out
    subj. features -> FC -> RELU -> FC -> RELU -> |
    """

    def __init__(self, num_classes, attention='multi_head',
                 use_language=True, use_spatial=True):
        """Initialize model."""
        super().__init__()
        self._attention_type = attention
        self.fc_subj = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc_obj = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc_subj_b = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.fc_obj_b = nn.Sequential(
            nn.Linear(1024, 256), nn.ReLU(), nn.Linear(256, 128), nn.ReLU())
        self.attention_layer = AttentionLayer(
            use_language and attention is not None,
            use_spatial and attention is not None)
        if attention == 'multi_head':
            self.classifier_weights = AttentionalWeights(
                num_classes, feature_dim=128, attention_type=attention)
            self.bias = nn.Parameter(torch.rand(1, num_classes))
            self.binary_classifier_weights = AttentionalWeights(
                2, feature_dim=128, attention_type=attention)
            self.binary_bias = nn.Parameter(torch.rand(1, 2))
        else:
            self.classifier_weights = nn.Linear(128, num_classes)
            self.binary_classifier_weights = nn.Linear(128, 2)

    def forward(self, subj_feats, obj_feats, subj_embs, obj_embs, masks,
                deltas):
        """Forward pass, return output scores."""
        attention = self.attention_layer(subj_embs, obj_embs, deltas, masks)
        os_feats = self.fc_obj(obj_feats) - self.fc_subj(subj_feats)
        os_feats = os_feats.unsqueeze(-1)
        bin_os_feats = self.fc_obj_b(obj_feats) - self.fc_subj_b(subj_feats)
        bin_os_feats = bin_os_feats.unsqueeze(-1)
        if self._attention_type == 'multi_head':
            return (
                torch.sum(
                    os_feats * self.classifier_weights(attention),
                    dim=1)
                + self.bias,
                torch.sum(
                    os_feats * self.binary_classifier_weights(attention),
                    dim=1)
                + self.binary_bias
            )
        return (
            self.classifier_weights(os_feats.view(-1, 128)),
            self.binary_classifier_weights(bin_os_feats.view(-1, 128))
        )


class AttentionalWeights(nn.Module):
    """Compute weights based on spatio-linguistic attention."""

    def __init__(self, num_classes, feature_dim, attention_type):
        """Initialize model."""
        super().__init__()
        self.num_classes = num_classes
        self.feat_dim = feature_dim
        self.attention_type = attention_type
        if attention_type == 'multi_head':
            self.att_fc = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, feature_dim * num_classes), nn.ReLU())
        elif attention_type == 'single_head':
            self.att_fc = nn.Sequential(
                nn.Linear(128, 64), nn.ReLU(),
                nn.Linear(64, feature_dim), nn.ReLU())

    def forward(self, attention):
        """Forward pass."""
        if self.attention_type is None:
            return None
        if self.attention_type == 'single_head':
            return self.att_fc(attention).view(-1, self.feat_dim, 1)
        return self.att_fc(attention).view(-1, self.feat_dim, self.num_classes)


class AttentionLayer(nn.Module):
    """Drive attention using language and/or spatial features."""

    def __init__(self, use_language=True, use_spatial=True):
        """Initialize model."""
        super().__init__()
        self._use_language = use_language
        self._use_spatial = use_spatial
        self.fc_subject = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_object = nn.Sequential(nn.Linear(300, 256), nn.ReLU())
        self.fc_lang = nn.Sequential(
            nn.Linear(512, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU())
        self.mask_net = nn.Sequential(
            nn.Conv2d(2, 32, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(32, 64, 5, stride=2, padding=2), nn.ReLU(),
            nn.Conv2d(64, 128, 8), nn.ReLU())
        self.fc_delta = nn.Sequential(
            nn.Linear(38, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())
        self.fc_spatial = nn.Sequential(
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU())

    def forward(self, subj_embs, obj_embs, deltas, masks):
        """Forward pass."""
        lang_attention, spatial_attention = None, None
        if self._use_language:
            lang_attention = self.fc_lang(torch.cat((
                self.fc_subject(subj_embs),
                self.fc_object(obj_embs)
            ), dim=1))
        if self._use_spatial:
            spatial_attention = self.fc_spatial(torch.cat((
                self.mask_net(masks).view(masks.shape[0], -1),
                self.fc_delta(deltas)
            ), dim=1))
        if self._use_language or self._use_spatial:
            attention = 0
            if self._use_language:
                attention = attention + lang_attention
            if self._use_spatial:
                attention = attention + spatial_attention
            return attention
        return None


class AttentionalPoolingLayer(nn.Module):
    """Attentional Pooling layer."""

    def __init__(self):
        """Initialize model."""
        super().__init__()
        self.register_buffer('const', torch.FloatTensor([0.0001]))
        self.softplus = nn.Softplus()

    def forward(self, features, weights):
        """
        Forward pass.

        Inputs:
            - features: tensor (batch_size, 256, 4, 4), the feature map
            - weights: tensor (batch, 256, num_classes),
                per-class attention weights
        """
        features = features.unsqueeze(-1)
        att_num = (  # (bs, 4, 4, num_classes)
            self.softplus(
                (features * weights.unsqueeze(2).unsqueeze(2)).sum(1))
            + self.const)
        att_denom = att_num.sum(2).sum(1)  # (bs, num_classes)
        attention_map = (  # (bs, 4, 4, num_classes)
            att_num
            / att_denom.unsqueeze(1).unsqueeze(2))
        return (attention_map.unsqueeze(1) * features).sum(3).sum(2)
