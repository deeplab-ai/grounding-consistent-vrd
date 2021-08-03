# -*- coding: utf-8 -*-
"""
A class for training/testing a network on Scene Graph Generation.

Methods _compute_loss and _net_outputs assume that _net_forward
returns (pred_scores, rank_scores, ...).
They should be re-implemented if that's not the case.
"""

import json
import random
from pdb import set_trace

import numpy as np
import torch
from torch.nn import functional as F
from tqdm import tqdm

from research.src.evaluators import (
    RankingClsEvaluator, RelationshipClsEvaluator, RelationshipEvaluator,
    NegativesEvaluator
)
from research.src.data_loaders import SGGDataset, SGGDataLoader
from common.tools import AnnotationLoader

from .base_train_tester_class import BaseTrainTester


class SGGTrainTester(BaseTrainTester):
    """
    Train and test utilities for Scene Graph Generation.

    Inputs upon initialization:
        - net: PyTorch nn.Module, the network to train/test
        - config: class Config, see config.py
        - features: set of str, features to load for net
        - obj_classifier: ObjectClassifier object (see corr. API)
        - teacher: a loaded SGG model
    """

    def __init__(self, net, config, features, obj_classifier=None,
                 teacher=None):
        """Initiliaze train/test instance."""
        super().__init__(net, config, features)
        self.obj_classifier = obj_classifier
        self.teacher = teacher

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        # Settings and loading
        self.logger.info(
            "Test %s on %s on %s" % (self._net_name, self._task, self._dataset)
        )
        self.training_mode = False
        self.net.eval()
        self.net.to(self._device)
        self._set_data_loaders(mode_ids={'test': 2})
        self.data_loader = self._data_loaders['test']
        if self._task == 'predcls' and self._use_multi_tasking:
            rank_eval = RankingClsEvaluator(self.annotation_loader)
        if self._test_on_negatives:
            rel_eval = NegativesEvaluator(self.annotation_loader,
                                          self._use_merged, self._misc_params.get('precision', 'fmP+'))
        elif self._compute_accuracy or self._dataset in {'VG80K', 'UnRel'}:
            rel_eval = RelationshipClsEvaluator(self.annotation_loader,
                                                self._use_merged)
        else:
            rel_eval = RelationshipEvaluator(self.annotation_loader,
                                             self._use_merged)

        # Forward pass on test set
        results = {}
        for batch in tqdm(self.data_loader):
            for step in range(len(batch['filenames'])):
                boxes = batch['boxes'][step]  # detected boxes
                labels = batch['labels'][step]  # store (s,p,o) labels here
                pred_scores, rank_scores, subj_scores, obj_scores = \
                    self._net_outputs(batch, step)
                scores = pred_scores.cpu().numpy()
                if self._task not in {'preddet', 'predcls'}:
                    subj_scores = subj_scores.cpu().numpy()
                    obj_scores = obj_scores.cpu().numpy()
                    scores = (
                            np.max(subj_scores, axis=1)[:, None]
                            * scores
                            * np.max(obj_scores, axis=1)[:, None])
                    labels[:, 0] = np.argmax(subj_scores, axis=1)
                    labels[:, 2] = np.argmax(obj_scores, axis=1)
                filename = batch['filenames'][step]
                if self._classes_to_keep is not None:
                    scores[:, self._classes_to_keep] += 10
                rel_eval.step(filename, scores, labels, boxes,
                              self._phrase_recall)
                # neg_eval.step(filename, scores)
                if rank_scores is not None and self._task == 'predcls':
                    rank_eval.step(filename, rank_scores.cpu().numpy())
                results.update({  # top-5 classification results
                    filename: {
                        'scores': np.sort(scores)[:, ::-1][:, :5].tolist(),
                        'classes': scores.argsort()[:, ::-1][:, :5].tolist()
                    }})
        # Print metrics and save results
        rel_eval.print_stats(self._task)
        if rank_scores is not None and self._task == 'predcls':
            rank_eval.print_stats()
        end = '_tail.json' if self._classes_to_keep is not None else '.json'
        if self._dataset not in self._net_name:
            end = end.replace('.', '_%s.' % self._dataset)
        if not self._test_on_negatives:
            with open(self._results_path + 'results' + end, 'w') as fid:
                json.dump(results, fid)

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        # Net outputs and targets
        outputs = self._net_forward(batch, step)
        scores = outputs[0]
        targets = self.data_loader.get('predicate_ids', batch, step)

        # Losses

        # Loss: CE
        losses = {'CE': self.criterion(scores, targets)}
        loss = losses['CE']

        # Knowledge Distillation
        if self._use_multi_tasking and self._task != 'preddet':
            loss += self._multitask_loss(outputs[1], batch, step)
        if self.teacher is not None and not self._use_consistency_loss:
            losses['KD'] = self._kd_loss(scores, outputs[1], batch, step)
            if self.training_mode:
                loss += losses['KD']

        # Consistency Loss
        if self._use_consistency_loss and self._epoch >= 1:
            losses['Cons'] = self._consistency_loss(batch, step, scores)
            loss += losses['Cons']

        # Graphical Contrastive Loss (Zhang 19)
        if self._use_graphl_loss:
            (
                losses['GraphL-L1'], losses['GraphL-L2'], losses['GraphL-L3']
            ) = self._graphical_contrastive_loss(batch, step, scores, targets)
            loss += (
                    1.0 * losses['GraphL-L1']
                    + 1.0 * losses['GraphL-L2']
                    + 1.0 * losses['GraphL-L3']
            )

        return loss, losses

    def _consistency_loss(self, batch, step, scores):
        """"""
        unlabeled = batch['predicate_ids'][step] == self.net.num_rel_classes - 1
        unlabeled = unlabeled.to(self._device)
        self.teacher.eval()
        self.teacher.mode = 'test'
        predicate_ids = scores.argmax(1)
        with torch.no_grad():
            t_outputs = self.teacher(
                self.data_loader.get('images', batch, step),
                self.data_loader.get('object_rois', batch, step),
                predicate_ids,
                self.data_loader.get('object_ids', batch, step),
                self.data_loader.get('pairs', batch, step),
                self.data_loader.get('image_info', batch, step)
            )
        boxes = self.data_loader.get('object_rois', batch, step)
        masks = self.teacher.get_obj_masks(boxes)
        pairs = batch['pairs'][step]
        subj_masks = masks[pairs[:, 0]].flatten(1)
        obj_masks = masks[pairs[:, 1]].flatten(1)
        # Calculate ground scores from teacher
        soft_scores = F.softmax(scores, dim=1)
        pred_scores = soft_scores.max(1)[0]
        subj_hmaps = t_outputs[0].flatten(1)
        obj_hmaps = t_outputs[1].flatten(1)
        subj_hmaps_norm = \
            subj_hmaps / (subj_hmaps.max(1)[0].unsqueeze(1) + 1e-8)
        obj_hmaps_norm = \
            obj_hmaps / (obj_hmaps.max(1)[0].unsqueeze(1) + 1e-8)
        subj_ground_scores = (subj_hmaps_norm * subj_masks).max(1)[0]
        obj_ground_scores = (obj_hmaps_norm * obj_masks).max(1)[0]
        ground_scores = (subj_ground_scores + obj_ground_scores) / 2
        loss_cons = torch.tensor([0.0]).to(self._device)
        if unlabeled.sum() > 0:
            loss_cons = F.binary_cross_entropy(
                pred_scores[unlabeled], ground_scores[unlabeled], reduction='none')
        return loss_cons.mean()


    def _negatives_loss(self, outputs, batch, step, neg_classes, typ):
        """
        Various Losses relative to Negatives
        - neg_classes: list, index of classes that loss is applied
        - typ: string
            train a ranker:
                - rank[_fair]
            train using a ranker as a teacher:
                - contrastive_(margin | log_sigmoid | bce | softmin)[_distil]
                - triplet
            use negative annotations from rules
                - negative
        """
        # FIX: add proper normalization
        scores = outputs[0]
        targets = self.data_loader.get('predicate_ids', batch, step)
        loss_contr = torch.as_tensor([0.0]).to(self._device)
        if typ == 'negative':
            neg_classes = self.data_loader.get('negative_ids', batch, step)
            has_neg = neg_classes != -1
            if has_neg.sum() == 0:
                return loss_contr
            scores_sm = torch.softmax(scores, dim=1)
            neg_scores = scores_sm[has_neg, neg_classes[has_neg]]
            loss_contr = -torch.log(1 - neg_scores).mean()
            return loss_contr
        self.teacher.eval()
        t_outputs = self.teacher(
            self.data_loader.get('images', batch, step),
            self.data_loader.get('object_rois', batch, step),
            self.data_loader.get('object_ids', batch, step),
            self.data_loader.get('pairs', batch, step),
            self.data_loader.get('image_info', batch, step)
        )
        t_scores = t_outputs[0]
        l_scores = F.softmax(t_outputs[0], dim=1) if 'rank' in typ \
            else F.softmax(t_outputs[2], dim=1)
        l_classes = l_scores.argmax(1)
        if 'distil' in typ:
            # use top classes of an other network
            # as positive class suggestions
            distil_scores = t_outputs[3]
        thres = 0.5
        for target_class in neg_classes:
            prior_indx = (l_scores.max(1)[0] > thres) * \
                         (l_classes == target_class)
            # UoI: Unlabeled of Interest
            # high_prior && unlabeled
            UoI = prior_indx * (targets == self.net.num_rel_classes - 1)
            if 'rank' in typ:
                for target_class in neg_classes:
                    pos_indx = prior_indx * (targets == target_class)
                    pos_scores = scores[pos_indx, target_class]
                    neg_scores = scores[UoI, target_class]
                    if 'fair' in typ:
                        limit_pos = limit_neg = min(
                            neg_scores.shape[0], pos_scores.shape[0]
                        )
                    else:
                        limit_pos = pos_scores.shape[0]
                        limit_neg = neg_scores.shape[0]
                    if limit_neg > 0:
                        rand_indx_neg = torch.multinomial(torch.ones_like(
                            neg_scores, dtype=torch.float), limit_neg)
                        neg_scores = neg_scores[rand_indx_neg]
                        neg_scores = F.sigmoid(neg_scores)
                        loss_contr += -torch.log(1 - neg_scores).mean() \
                                      / ((limit_pos > 0) + 1)
                    if limit_pos > 0:
                        rand_indx_pos = torch.multinomial(torch.ones_like(
                            pos_scores, dtype=torch.float), limit_pos)
                        pos_scores = pos_scores[rand_indx_pos]
                        pos_scores = F.sigmoid(pos_scores)
                        loss_contr += -torch.log(pos_scores).mean() \
                                      / ((limit_neg > 0) + 1)
            elif 'contrastive' in typ:
                # UoI && teacher(Model trained with ranking-loss)
                # gives negative score
                neg_indx = UoI * (t_scores[:, target_class] < 0)
                if neg_indx.sum() > 0:
                    neg_scores = scores[neg_indx, target_class]
                    if 'distil' in typ:
                        pos_class = distil_scores.argmax(1)[neg_indx]
                    else:  # pick randomly positive classes
                        pos_class = torch.as_tensor([
                            random.choice(list(
                                set(range(scores.shape[1])) - {target_class}
                            ))
                            for i in range(neg_indx.sum())
                        ])
                    pos_scores = scores[neg_indx, pos_class]
                    scales = F.tanh(-t_scores[neg_indx, target_class])
                    if 'margin' in typ:
                        margin = 2
                        loss_contr += (
                                scales * torch.clamp(
                            neg_scores - pos_scores + margin, min=0)
                        ).mean()
                    elif 'log_sigmoid' in typ:
                        loss_contr += -(
                                F.logsigmoid(pos_scores - neg_scores) * scales
                        ).mean()
                    elif 'bce' in typ:
                        loss_contr += -torch.log(
                            1 - F.sigmoid(neg_scores)).mean()
                    elif 'softmin' in typ:
                        neg_targets = torch.ones(neg_indx.sum()) * target_class
                        neg_targets = neg_targets.long().to(self._device)
                        loss_contr += F.cross_entropy(
                            -scores[neg_indx], neg_targets,
                            ignore_index=self.net.num_rel_classes - 1
                        )
            elif 'triplet' in typ:
                # NOTE: add option to take pos/neg from loader
                neg_indx = UoI * (t_scores[:, target_class] < 0)
                pos_indx = prior_indx * (t_scores[:, target_class] >= 0)
                if (neg_indx.sum() > 0) and (pos_indx.sum() > 0):
                    pos_scores = scores[pos_indx, target_class].unsqueeze(1)
                    neg_scores = scores[neg_indx, target_class].unsqueeze(0)
                    neg_scores = neg_scores.expand(pos_indx.sum(), -1)
                    comb_scores = torch.cat((pos_scores, neg_scores), dim=1)
                    zero_targets = torch.zeros(
                        comb_scores.shape[0], dtype=torch.long
                    ).to(self._device)
                    loss_contr += F.cross_entropy(comb_scores, zero_targets)
        return loss_contr

    def _kd_loss(self, scores, bg_scores, batch, step):
        """Compute knowledge distillation loss."""
        unlabeled = batch['predicate_ids'][step] == self.net.num_rel_classes - 1
        if unlabeled.sum() == 0:
            return torch.tensor([0.0]).to(self._device)
        self.teacher.eval()
        with torch.no_grad():
            t_outputs = self.teacher(
                self.data_loader.get('images', batch, step),
                self.data_loader.get('object_rois', batch, step),
                self.data_loader.get('object_ids', batch, step),
                self.data_loader.get('pairs', batch, step),
                self.data_loader.get('image_info', batch, step)
            )
        kd_loss = 80 * F.kl_div(
            F.log_softmax(scores[unlabeled], 1), F.softmax(t_outputs[0][unlabeled], 1),
            reduction='none'
        ).mean(1).mean(0)
        if self._use_multi_tasking and self._task != 'preddet':
            kd_loss += 5 * F.kl_div(
                F.log_softmax(bg_scores, 1), F.softmax(t_outputs[1], 1),
                reduction='none'
            ).mean(1)
        return kd_loss

    def _multitask_loss(self, bg_scores, batch, step):
        """Reformulate loss to involve bg/fg multi-tasking."""
        bg_targets = self.data_loader.get('bg_targets', batch, step)
        return F.cross_entropy(bg_scores, bg_targets, reduction='none')

    def _graphical_contrastive_loss(self, batch, step, scores, targets):
        pairs = self.data_loader.get('pairs', batch, step)
        obj_ids = self.data_loader.get('object_ids', batch, step)
        l1 = (
                self.spo_aggnostic_loss(scores, pairs, targets, obj_ids)
                + self.spo_aggnostic_loss(scores, pairs[:, [1, 0]], targets,
                                          obj_ids)
        )
        l2 = (
                self.so_aware_loss(scores, pairs, targets, obj_ids)
                + self.so_aware_loss(scores, pairs[:, [1, 0]], targets,
                                     obj_ids)
        )
        l3 = (
                self.p_aware_loss(scores, pairs, targets, obj_ids)
                + self.p_aware_loss(scores, pairs[:, [1, 0]], targets, obj_ids)
        )
        return l1, l2, l3

    def _net_forward(self, batch, step):
        """Return a tuple of scene graph generator's outputs."""
        return self.net(
            self.data_loader.get('images', batch, step),
            self.data_loader.get('object_rois', batch, step),
            self.data_loader.get('object_ids', batch, step),
            self.data_loader.get('pairs', batch, step),
            self.data_loader.get('image_info', batch, step)
        )

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch."""
        obj_vecs = self._object_forward(batch, step)
        pairs = self.data_loader.get('pairs', batch, step)
        s_scores = obj_vecs[pairs[:, 0]]
        o_scores = obj_vecs[pairs[:, 1]]
        rest_outputs = self._net_forward(batch, step)
        p_scores = rest_outputs[0]
        if self._task == 'preddet' or not self._use_multi_tasking:
            return p_scores, None, s_scores, o_scores
        bg_scores = rest_outputs[1]
        return (
            p_scores * bg_scores[:, 1].unsqueeze(-1),
            bg_scores, s_scores, o_scores
        )

    def _object_forward(self, batch, step, base_feats=None):
        """Return object vectors for different tasks."""
        if self.net.mode == 'train' or self._task != 'sgcls':
            obj_ids = self.data_loader.get('object_ids', batch, step)
            obj_vecs = torch.zeros(len(obj_ids), self._num_obj_classes)
            obj_vecs = obj_vecs.to(self._device)
            obj_vecs[torch.arange(len(obj_ids)), obj_ids] = 1.0
            if self._task == 'sggen':
                # multiply with obj-cls prob
                obj_scores = self.data_loader.get('object_scores', batch, step)
                obj_vecs *= obj_scores[:, None]
        elif self._task == 'sgcls':
            obj_vecs = self.obj_classifier(
                self.data_loader.get('images', batch, step),
                self.data_loader.get('object_rois', batch, step),
                self.data_loader.get('object_ids', batch, step),
                self.data_loader.get('image_info', batch, step)
            )
        return obj_vecs

    def _set_data_loaders(self, mode_ids={'train': 0, 'val': 1, 'test': 2}):
        annotations = np.array(self.annotation_loader.get_annos())
        split_ids = np.array([anno['split_id'] for anno in annotations])
        datasets = {
            split: SGGDataset(
                annotations[split_ids == split_id].tolist(),
                self.config, self.features)
            for split, split_id in mode_ids.items()
        }
        self._data_loaders = {
            split: SGGDataLoader(
                datasets[split], batch_size=self._batch_size,
                shuffle=split == 'train', num_workers=self._num_workers,
                drop_last=split != 'test', device=self._device)
            for split in mode_ids
        }
        if 'overfit' in mode_ids:
            random.seed(0)
            overfit_idx = random.sample(list(np.where(split_ids == 0)[0]), 20)
            datasets['overfit'] = SGGDataset(
                annotations[overfit_idx].tolist(),
                self.config, self.features)
            self._data_loaders['overfit'] = SGGDataLoader(
                datasets['overfit'], batch_size=1,
                shuffle=True, num_workers=self._num_workers,
                drop_last=False, device=self._device)

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
