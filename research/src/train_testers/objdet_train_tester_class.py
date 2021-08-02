# -*- coding: utf-8 -*-
"""
Functions for training and testing an object detector.

Additionally to model evaluation, the test method constructs a
json of annotations, in order to be used for the SGGen task.
"""

from collections import defaultdict
import json

import numpy as np
import torch
from tqdm import tqdm

from research.src.evaluators import ObjectDetEvaluator
from research.src.data_loaders import ObjDataset, SGGDataLoader
from .base_train_tester_class import BaseTrainTester


class ObjDetTrainTester(BaseTrainTester):
    """Train and test utilities for Object Detection."""

    def __init__(self, net, config):
        """Initiliaze train/test instance."""
        super().__init__(net, config, set())

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        # Settings and loading
        self.logger.info(
            "Test %s on objdet on %s" % (self._net_name, self._dataset))
        self.net.eval()
        self.net.to(self._device)
        self._set_data_loaders(mode_ids={'test': 2})
        self.data_loader = self._data_loaders['test']
        obj_eval = ObjectDetEvaluator(self.annotation_loader)
        anno_constructor = SGGenAnnosConstructor(self._dataset, self.config)
        max_per_img = self.config.max_obj_dets_per_img

        # Forward pass on test set, epoch=0
        for batch in tqdm(self.data_loader):
            for step in range(len(batch['filenames'])):
                scores, bboxes, labels = self._net_outputs(batch, step)
                scores = scores.cpu().numpy()
                bboxes = bboxes.cpu().numpy().astype(int)
                labels = labels.cpu().numpy()
                score_sort = scores.argsort()[::-1][:max_per_img]
                labels = labels[score_sort]
                bboxes = bboxes[score_sort]
                scores = scores[score_sort]
                # boxes: (x1, y1, x2, y2) back to (y1, y2, x1, x2)
                bboxes = bboxes[:, (1, 3, 0, 2)]
                filename = batch['filenames'][step]
                obj_eval.step(filename, scores, bboxes, labels)
                anno_constructor.step(filename, scores, bboxes, labels)
        # Print metrics and save results
        obj_eval.print_stats()
        anno_constructor.save()

    def _compute_train_loss(self, batch):
        """Compute train loss."""
        accum_loss = [
            self._compute_loss(batch, step)[0]
            for step in range(len(batch['filenames']))
        ]
        return sum(accum_loss) / len(accum_loss), {}

    @torch.no_grad()
    def _compute_validation_loss(self):
        """Compute validation loss."""
        self.training_mode = False
        # avoid self.net.eval() to get losses
        self.data_loader = self._data_loaders['val']
        losses = [
            self._compute_loss(batch, step)
            for batch in self.data_loader
            for step in range(len(batch['filenames']))
        ]
        loss = sum([l[0].item() for l in losses]) / len(losses)
        print_loss = defaultdict(int)
        for _, loss_dict in losses:
            for key, val in loss_dict.items():
                print_loss[key] += val.item()
        for key, val in print_loss.items():
            print_loss[key] = val / len(losses)
        self.net.train()
        self.data_loader = self._data_loaders['train']
        self.training_mode = True
        return loss, print_loss

    def _compute_loss(self, batch, step):
        """Compute loss for current batch."""
        losses = self._net_forward(batch, step)
        loss = (
            losses['loss_classifier'] + losses['loss_box_reg']
            + losses['loss_objectness'] + losses['loss_rpn_box_reg']
        )
        return loss, losses

    def _net_forward(self, batch, step):
        return self.net(
            self.data_loader.get('images', batch, step),
            {
                'boxes': self.data_loader.get('object_rois', batch, step),
                'labels': self.data_loader.get('object_ids', batch, step)
            }
        )

    def _net_outputs(self, batch, step):
        """Get network outputs for current batch."""
        outputs = self._net_forward(batch, step)
        return outputs[0]['scores'], outputs[0]['boxes'], outputs[0]['labels']

    def _set_data_loaders(self, mode_ids={'train': 0, 'val': 1, 'test': 2}):
        annotations = np.array(self.annotation_loader.get_annos())
        split_ids = np.array([anno['split_id'] for anno in annotations])
        datasets = {
            split: ObjDataset(
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

    def _setup_optimizer(self):
        logit_params = [  # extra parameters, train with config lr
            param for name, param in self.net.named_parameters()
            if 'box_head' not in name and 'rpn' not in name
            and param.requires_grad]
        backbone_params = [  # backbone net parameters, train with lower lr
            param for name, param in self.net.named_parameters()
            if 'box_head' in name or 'rpn' in name
            and param.requires_grad]
        return torch.optim.Adam(
            [
                {'params': backbone_params, 'lr': 0.1 * self._learning_rate},
                {'params': logit_params}
            ], lr=self._learning_rate, weight_decay=self._weight_decay)


class SGGenAnnosConstructor:
    """Create SGGen annotations with detected boxes."""

    def __init__(self, dataset, config):
        """Load dataset and keep test annotations."""
        self._dataset = dataset
        self._num_classes = config.num_rel_classes
        self._cls_names = np.array(config.obj_classes)
        self._json_path = config.paths['json_path']
        # TODO: sggen annos should use predcls pairs for training
        with open(self._json_path + dataset + '_preddet.json') as fid:
            annos = json.load(fid)
        self._annos = {anno['filename']: anno for anno in annos}

    def step(self, filename, scores, bboxes, labels):
        """Save a new image annotation."""
        if filename in self._annos:
            anno = dict(self._annos[filename])
            anno['objects'] = {
                'ids': labels.tolist(),
                'boxes': bboxes.tolist(),
                'names': self._cls_names[labels.astype(int)].tolist(),
                'scores': scores.tolist()
            }
            subj_ids, obj_ids = self._create_all_pairs(len(scores))
            anno['relations'] = {
                'ids': [self._num_classes - 1] * len(subj_ids),
                'names': ['__background__'] * len(subj_ids),
                'subj_ids': subj_ids.tolist(),
                'obj_ids': obj_ids.tolist()
            }
            self._annos[filename] = dict(anno)

    def save(self):
        """Save annotations to file."""
        with open(self._json_path + self._dataset + '_sggen.json', 'w') as fid:
            json.dump(list(self._annos.values()), fid)

    @staticmethod
    def _create_all_pairs(num_objects):
        """Create all possible combinations of objects."""
        obj_inds = np.arange(num_objects)
        return np.where(obj_inds[:, None] != obj_inds.T[None, :])
