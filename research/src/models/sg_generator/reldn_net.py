# -*- coding: utf-8 -*-
"""Relationship Detection Network by Zhang et al., 2019."""

import torch
import torch.nn.functional as F

from common.models.sg_generator import RelDN
from research.src.train_testers import SGGTrainTester

class TrainTester(SGGTrainTester):
    """Extends SGGTrainTester."""

    def __init__(self, net, config, features, obj_classifier, teacher):
        """Initialize instance."""
        super().__init__(net, config, features, obj_classifier, teacher)

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self._net_forward(batch, step)
        scores = outputs[0]
        vis_scores, spat_scores = outputs[2:4]
        targets = self.data_loader.get('predicate_ids', batch, step)
        pairs = self.data_loader.get('pairs', batch, step)
        obj_ids = self.data_loader.get('object_ids', batch, step)

        # Losses
        losses = {
            'CE': self.criterion(scores, targets),
            'vis-CE': self.criterion(vis_scores, targets),
            'spat-CE': self.criterion(spat_scores, targets),
            'L1': (
                self.spo_aggnostic_loss(scores, pairs, targets, obj_ids)
                + self.spo_aggnostic_loss(scores, pairs[:, [1, 0]], targets,
                                          obj_ids)
            ),
            'L2': (
                self.so_aware_loss(scores, pairs, targets, obj_ids)
                + self.so_aware_loss(scores, pairs[:, [1, 0]], targets,
                                     obj_ids)
            ),
            'L3': (
                self.p_aware_loss(scores, pairs, targets, obj_ids)
                + self.p_aware_loss(scores, pairs[:, [1, 0]], targets, obj_ids)
            )

        }

        loss = (
            losses['CE'] + losses['vis-CE'] + losses['spat-CE']
        )
        if self._use_multi_tasking and self._task != 'preddet':
            loss += self._multitask_loss(outputs[1], batch, step)
        if self.teacher is not None and not self._use_consistency_loss:
            losses['KD'] = self._kd_loss(scores, outputs[1], batch, step)
            if self.training_mode:
                loss += losses['KD']
        if self._use_consistency_loss and self._epoch >= 1:
            cons_loss = \
                self._consistency_loss(batch, step, scores, typ='triplet_sm')
            losses['Cons'] = cons_loss
            if self.training_mode:
                loss += cons_loss
        return loss, losses

    def spo_aggnostic_loss(self, scores, pairs, targets, obj_ids, margin=0.2):
        pairs = torch.tensor(pairs)
        # rel_prob = torch.softmax(scores, dim=1)[:, :-1].sum(1)
        rel_prob = 1 - torch.softmax(scores, dim=1)[:, -1]
        pos_batch = []
        neg_batch = []
        for i in range(len(obj_ids)):
            inds = torch.where(pairs[:, 0] == i)[0]
            pos_inds = inds[targets[inds] != self.net.num_rel_classes - 1]
            neg_inds = inds[targets[inds] == self.net.num_rel_classes - 1]
            if pos_inds.nelement() == 0 or neg_inds.nelement() == 0:
                continue
            pos_batch.append(torch.min(rel_prob[pos_inds]))
            neg_batch.append(torch.max(rel_prob[neg_inds]))
        if len(neg_batch) == 0:
            return torch.zeros(1).to(self._device)
        pos_batch = (torch.stack(pos_batch) if len(pos_batch)
                     else torch.tensor([]).to(self._device))
        neg_batch = (torch.stack(neg_batch) if len(neg_batch)
                     else torch.tensor([]).to(self._device))
        y = torch.ones_like(pos_batch)
        return F.margin_ranking_loss(pos_batch, neg_batch, y, margin=margin)

    def so_aware_loss(self, scores, pairs, targets, obj_ids, margin=0.2):
        pairs = torch.tensor(pairs)
        # rel_prob = torch.softmax(scores, dim=1)[:, :-1].sum(1)
        rel_prob = 1 - torch.softmax(scores, dim=1)[:, -1]
        pos_batch = []
        neg_batch = []
        for i in range(len(obj_ids)):
            inds = torch.where(pairs[:, 0] == i)[0]
            pos_inds = inds[targets[inds] != self.net.num_rel_classes - 1]
            neg_inds = inds[targets[inds] == self.net.num_rel_classes - 1]
            if pos_inds.nelement() == 0 or neg_inds.nelement() == 0:
                continue
            for j in torch.unique(obj_ids[pairs[pos_inds, 1]]):
                pos_inds_j = pos_inds[obj_ids[pairs[pos_inds, 1]] == j]
                neg_inds_j = neg_inds[obj_ids[pairs[neg_inds, 1]] == j]
                if neg_inds_j.nelement() == 0:
                    continue
                pos_batch.append(torch.min(rel_prob[pos_inds_j]))
                neg_batch.append(torch.max(rel_prob[neg_inds_j]))
        if len(neg_batch) == 0:
            return torch.zeros(1).to(self._device)
        pos_batch = (torch.stack(pos_batch) if len(pos_batch)
                     else torch.tensor([]).to(self._device))
        neg_batch = (torch.stack(neg_batch) if len(neg_batch)
                     else torch.tensor([]).to(self._device))
        y = torch.ones_like(pos_batch)
        return F.margin_ranking_loss(pos_batch, neg_batch, y, margin=margin)

    def p_aware_loss(self, scores, pairs, targets, obj_ids, margin=0.2):
        pairs = torch.tensor(pairs)
        # rel_prob = torch.softmax(scores, dim=1)[:, :-1].sum(1)
        rel_prob = 1 - torch.softmax(scores, dim=1)[:, -1]
        rel_pred = scores[:, :-1].max(1)[1]
        pos_batch = []
        neg_batch = []
        for i in range(len(obj_ids)):
            inds = torch.where(pairs[:, 0] == i)[0]
            pos_inds = inds[targets[inds] != self.net.num_rel_classes - 1]
            neg_inds = inds[targets[inds] == self.net.num_rel_classes - 1]
            if pos_inds.nelement() == 0 or neg_inds.nelement() == 0:
                continue
            for j in torch.unique(targets[pos_inds]):
                pos_inds_j = pos_inds[targets[pos_inds] == j]
                neg_inds_j = neg_inds[rel_pred[neg_inds] == j]
                if neg_inds_j.nelement() == 0:
                    continue
                pos_batch.append(torch.min(rel_prob[pos_inds_j]))
                neg_batch.append(torch.max(rel_prob[neg_inds_j]))
        if len(neg_batch) == 0:
            return torch.zeros(1).to(self._device)
        pos_batch = (torch.stack(pos_batch) if len(pos_batch)
                     else torch.tensor([]).to(self._device))
        neg_batch = (torch.stack(neg_batch) if len(neg_batch)
                     else torch.tensor([]).to(self._device))
        y = torch.ones_like(pos_batch)
        return F.margin_ranking_loss(pos_batch, neg_batch, y, margin=margin)


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a net."""
    net = RelDN(config)
    train_tester = TrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()
