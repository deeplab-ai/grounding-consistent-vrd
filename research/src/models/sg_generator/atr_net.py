# -*- coding: utf-8 -*-
"""Attention-Translation-Relation Network, Gkanatsios et al., 2019."""

import torch
from torch.nn import functional as F

from common.models.sg_generator import ATRNet
from research.src.train_testers import SGGTrainTester


class TrainTester(SGGTrainTester):
    """Extends SGGTrainTester."""

    def __init__(self, net, config, features, obj_classifier, teacher):
        """Initialize instance."""
        super().__init__(net, config, features, obj_classifier, teacher)
        self.batch_counter = 0

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self._net_forward(batch, step)
        scores, p_scores, os_scores = (outputs[0], outputs[2], outputs[3])
        targets = self.data_loader.get('predicate_ids', batch, step)

        # Losses
        # targets_1hot = torch.zeros_like(scores)
        # targets_1hot[torch.arange(len(scores)), targets] = 1
        # losses = {'CE': F.binary_cross_entropy_with_logits(scores, targets_1hot, reduction='none').mean(1),
        #           'p-CE': F.binary_cross_entropy_with_logits(p_scores, targets_1hot, reduction='none').mean(1),
        #           'os-CE': F.binary_cross_entropy_with_logits(os_scores, targets_1hot, reduction='none').mean(1)
        #           }
        losses = {
            'CE': self.criterion(scores, targets),
            'p-CE': self.criterion(p_scores, targets),
            'os-CE': self.criterion(os_scores, targets)
        }
        loss = losses['CE'] + losses['p-CE'] + losses['os-CE']
        if self._use_multi_tasking and self._task != 'preddet':
            bg_scores = outputs[1]
            bg_p_scores = outputs[4]
            bg_os_scores = outputs[5]
            bg_targets = self.data_loader.get('bg_targets', batch, step)
            loss = (
                    loss
                    + F.cross_entropy(bg_scores, bg_targets, reduction='none')
                    + F.cross_entropy(bg_p_scores, bg_targets, reduction='none')
                    + F.cross_entropy(bg_os_scores, bg_targets, reduction='none')
            )
        if self.teacher is not None and self._negative_loss is None \
                and not self._use_consistency_loss:
            losses['KD'] = self._kd_loss(scores, outputs[1], batch, step)
            if self.training_mode:
                loss += losses['KD']
        if self._negative_loss is not None:
            if self._neg_classes is not None:
                neg_classes = self._neg_classes
            else:
                neg_classes = [i for i in range(self.net.num_rel_classes - 1)]
            losses['NEG'] = self._negatives_loss(
                outputs, batch, step, neg_classes, self._negative_loss)
            if self.training_mode:
                loss += losses['NEG']
        if self._use_consistency_loss and self._epoch >= 1:
            cons_loss = \
                self._consistency_loss(batch, step, scores, typ='triplet_sm')
            losses['Cons'] = cons_loss
            if self.training_mode:
                loss += cons_loss
        return loss, losses


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a net."""
    net_params = config_net_params(config)
    net = ATRNet(
        config,
        attention=net_params['attention'],
        use_language=net_params['use_language'],
        use_spatial=net_params['use_spatial'])
    train_tester = TrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()


def config_net_params(config):
    """Configure net parameters."""
    net_params = {
        'attention': 'multi_head',
        'use_language': True,
        'use_spatial': True
    }
    if 'single_head' in config.net_name:
        net_params['attention'] = 'single_head'
    if 'no_att' in config.net_name:
        net_params['attention'] = None
    if 'no_lang' in config.net_name:
        net_params['use_language'] = False
    if 'no_spat' in config.net_name:
        net_params['use_spatial'] = False
    return net_params
