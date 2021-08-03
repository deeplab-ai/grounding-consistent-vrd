# -*- coding: utf-8 -*-
"""Union Visual Translation Embeddings net by Hung et al., 2019."""

import torch
from torch.nn import functional as F

from common.models.sg_generator import UVTransE
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
        v_scores, l_scores, subj_feats, pred_feats, obj_feats = outputs[2:]
        targets = self.data_loader.get('predicate_ids', batch, step)

        # Losses
        losses = {
            'v-CE': self.criterion(v_scores, targets),
            'l-CE': self.criterion(l_scores, targets),
            'pred': F.relu(pred_feats ** 2 - 1).mean(1),
            'obj': F.relu(obj_feats ** 2 - 1).mean(1),
            'subj': F.relu(subj_feats ** 2 - 1).mean(1)
        }
        loss = (
            0.5 * losses['v-CE'] + 0.5 * losses['l-CE']
            + losses['subj'] + losses['pred'] + losses['obj']
        )
        if self._use_multi_tasking and self._task != 'preddet':
            loss += self._multitask_loss(outputs[1], batch, step)
        if self.teacher is not None and not self._use_consistency_loss:
            losses['KD'] = (
                self._kd_loss(v_scores, outputs[1], batch, step)
                + self._kd_loss(l_scores, outputs[1], batch, step)
            )
            if self.training_mode:
                loss += losses['KD']
        if self._use_consistency_loss and self._epoch >= 1:
            cons_loss_vis = \
                self._consistency_loss(batch, step, v_scores, typ='triplet_sm')
            losses['Cons-vis'] = cons_loss_vis
            cons_loss_lang = \
                self._consistency_loss(batch, step, l_scores, typ='triplet_sm')
            losses['Cons-lang'] = cons_loss_lang
            if self.training_mode:
                loss += cons_loss_vis + cons_loss_lang
        return loss, losses


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a net."""
    net = UVTransE(config)
    train_tester = TrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()
