# -*- coding: utf-8 -*-
"""
Basic class for training/testing a projector on Scene Graph Generation.

Methods _compute_loss and _net_outputs assume that _net_forward
returns (pred_scores, rank_scores, classifiers ...).
They should be re-implemented if that's not the case.
"""

import torch
from torch.nn import functional as F

from common.tools import AnnotationLoader
from .sgg_train_tester_class import SGGTrainTester


class SGProjTrainTester(SGGTrainTester):
    """
    Train and test utilities for projectors on Scene Graph Generation.

    Inputs upon initialization:
        - net: PyTorch nn.Module, the network to train/test
        - config: class Config, see config.py
        - features: set of str, features to load for net
        - obj_classifier: ObjectClassifier object (see corr. API)
        - teacher: a loaded SGP model
    """

    def __init__(self, net, config, features, obj_classifier=None,
                 teacher=None):
        """Initiliaze train/test instance."""
        super().__init__(net, config, features, obj_classifier, teacher=teacher)
        self.features = features.union({'predicate_similarities'})

    def train_test(self):
        """Train and test a net, general handler for all tasks."""
        self.logger.debug(
            'Tackling %s for %d classes' % (self._task, self._num_classes))
        self.annotation_loader = AnnotationLoader(self.config)
        self.criterion = self._setup_loss_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler(self.optimizer)
        self.train()
        self.config.reset()
        self.net.reset_from_config(self.config)
        self._set_from_config(self.config)
        self.annotation_loader = AnnotationLoader(self.config)
        self.features.remove('predicate_similarities')
        self.net.mode = 'test'
        self.test()
        self.logger.info('Test complete')

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self._net_forward(batch, step)
        scores = outputs[0]
        classifiers = outputs[2]
        targets = self.data_loader.get('predicate_ids', batch, step)
        gt_sims = self.data_loader.get('predicate_similarities', batch, step)

        # Loss: CE
        losses = {'CE': self.criterion(5 * scores, targets)}

        # Loss: language synonymy MSE
        sims = F.cosine_similarity(  # predicate similarities given context!
            classifiers,
            classifiers[torch.arange(len(targets)), targets, :].unsqueeze(1),
            dim=2
        )
        to_consider = (gt_sims > -1).float()
        losses['sims'] = 50 * F.mse_loss(
            sims * to_consider,
            gt_sims * to_consider
        )

        # Loss: visual synonymy kl-divergence
        losses['kl'] = 1000 * F.kl_div(
            F.log_softmax(scores, 1),
            F.softmax(sims, 1),
            reduction='none'
        ).mean(1)

        # Total loss
        loss = (
            losses['CE'] * (0.5 if self._epoch > 3 else 1)
            + losses['kl'] * (1 if self._epoch > 3 else 0.1)
            + losses['sims']
        )

        # Multi-task Loss
        if self._use_multi_tasking and self._task != 'preddet':
            loss += self._multitask_loss(outputs[1], batch, step)

        # KD Loss
        if self.teacher is not None and self._negative_loss is None \
                and not self._use_consistency_loss:
            losses['KD'] = self._kd_loss(scores, outputs[1], batch, step)
            if self.training_mode:
                loss += losses['KD']

        # Consistency Loss
        if self._use_consistency_loss and self._epoch >= 0:
            losses['Cons'] = self._consistency_loss(batch, step, 5 * scores)
            if self.training_mode and self._epoch >= 0:
                loss += losses['Cons']
        return loss, losses