# -*- coding: utf-8 -*-
"""
Basic class for training/testing a network.

Each TrainTester class is useful for train and evaluating models
on the respective problem (SGG, ObjCls, ObjDet etc.).
"""

from collections import defaultdict
import json
from os import path as osp

import numpy as np
import torch
from torch.optim.lr_scheduler import MultiStepLR
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import matplotlib.pyplot as plt

from common.tools import EarlyStopping
from common.tools import AnnotationLoader


class BaseTrainTester:
    """
    Basic train and test utilities for multiple tasks.

    Inputs upon initialization:
        - net: PyTorch nn.Module, the network to train/test
        - config: class Config, see config.py
        - features: set of str, features to load for net
    """

    def __init__(self, net, config, features):
        """Initiliaze train/test instance."""
        self.net = net
        self.config = config
        self.features = features
        self._set_from_config(config)
        # Used for logging:
        self._steps = 0
        self._train_logs = 0
        self._val_logs = 0

    def _set_from_config(self, config):
        """Load config variables."""
        self._batch_size = config.batch_size
        self._classes_to_keep = config.classes_to_keep
        self._compute_accuracy = config.compute_accuracy
        self._dataset = config.dataset
        self._device = config.device
        self._epochs = config.epochs
        self._learning_rate = config.learning_rate
        self._models_path = config.paths['models_path']
        self._net_name = config.net_name
        self._num_classes = config.num_classes
        self._num_obj_classes = config.num_obj_classes
        self._num_workers = config.num_workers
        self._phrase_recall = config.phrase_recall
        self._patience = config.patience
        self._restore_on_plateau = config.restore_on_plateau
        self._results_path = config.paths['results_path']
        self._task = config.task
        self._test_dataset = config.test_dataset
        self._use_early_stopping = config.use_early_stopping
        self._use_merged = config.use_merged
        self._use_multi_tasking = config.use_multi_tasking
        self._use_negative_samples = config.use_negative_samples
        self._negative_loss = config.negative_loss
        self._neg_classes = config.neg_classes
        self._test_on_negatives = config.test_on_negatives
        self._use_weighted_ce = config.use_weighted_ce
        self._weight_decay = config.weight_decay
        self.logger = config.logger
        self.writer = SummaryWriter(config.logdir)
        self._use_consistency_loss = config.use_consistency_loss
        self._use_graphl_loss = config.use_graphl_loss
        self._misc_params = config.misc_params

    def train_test(self):
        """Train and test a net, general handler for all tasks."""
        self.logger.debug(
            'Tackling %s for %d classes' % (self._task, self._num_classes))
        self.annotation_loader = AnnotationLoader(self.config)
        self.criterion = self._setup_loss_criterion()
        self.optimizer = self._setup_optimizer()
        self.scheduler = self._setup_scheduler(self.optimizer)
        self.train()
        self.net.mode = 'test'
        self.test()
        self.logger.info('Test complete')

    def train(self):
        """Train a neural network if it does not already exist."""
        self.logger.info("Performing training for %s" % self._net_name)
        self.training_mode = True

        # Check for existent checkpoint
        model_path_name = osp.join(self._models_path, 'model.pt')
        trained, epoch = self._check_for_checkpoint(model_path_name)
        if trained:  # model is already trained
            return self.net
        epochs = list(range(self._epochs))[epoch:]

        # Settings and loading
        self.net.train()
        self.net.to(self._device)
        self._set_data_loaders(mode_ids={'train': 0, 'val': 1})
        self.data_loader = self._data_loaders['train']
        self.logger.debug("Batch size is %d", self._batch_size)

        # Main training procedure
        for epoch in epochs:
            keep_training = self._train_epoch(epoch, model_path_name)
            if not keep_training:
                self.logger.info('Model converged, exit training')
                break

        # Training is complete, save model
        self._save_model(model_path_name, epoch, True)
        self.logger.info('Finished Training')
        return self.net

    def _train_epoch(self, epoch, model_path_name):
        """Train the network for one epoch."""
        keep_training = True
        self._epoch = epoch
        self.logger.info('[Epoch %d]' % epoch)

        # Adjust learning rate
        if self.scheduler is not None and not self._use_early_stopping:
            self.scheduler.step()
        # NOTE: this shouldn't be needed
        for param_group in self.optimizer.param_groups:
            param_group['base_lr'] = param_group['lr']
        curr_lr = max(p['lr'] for p in self.optimizer.param_groups)
        self.logger.debug("Learning rate is now %f" % curr_lr)

        # Main epoch pipeline
        for batch in tqdm(self.data_loader):
            # Zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward + Backward + Optimize on batch data
            loss, losses = self._compute_train_loss(batch)
            loss.backward()
            self.optimizer.step()
            losses['Total'] = loss.item()
            self._steps += 1
            if self._steps % 50 == 0:
                self.writer.add_scalars('Train Loss', losses,
                                        self._train_logs)
                self._train_logs += 1

        # data logging for consistency_loss debugging, delete after usage
        # for k, v in self.accum_data.items():
        #     self.log_data[k].append(torch.cat(v).squeeze().tolist())
        #     self.accum_data[k] = []
        # self.log_data['weights'].append(self.net.ground_bias.detach().cpu().tolist())

        # After each epoch: check validation loss and convergence
        val_loss, val_losses = self._compute_validation_loss()
        self.writer.add_scalars('Val Loss', val_losses, self._val_logs)
        self._val_logs += 1
        improved = True
        if self._use_early_stopping and self.scheduler is not None:
            improved, ret_epoch, keep_training = self.scheduler.step(val_loss)
            if ret_epoch < epoch:
                if self._restore_on_plateau or not keep_training:
                    self._load_model(model_path_name,
                                     restore={'net', 'optimizer'})
                    self.scheduler.reduce_lr()
        # Save model
        if improved:
            self._save_model(model_path_name, epoch, not keep_training)
        return keep_training

    @torch.no_grad()
    def test(self):
        """Test a neural network."""
        self.logger.info(
            "Test %s on %s on %s" % (self._net_name, self._task, self._dataset)
        )
        self.training_mode = False

    def _check_for_checkpoint(self, model_path_name):
        """Check if a checkpoint exists."""
        if osp.exists(model_path_name):
            epoch, finished = self._load_model(model_path_name)
            # Trained model found
            if finished:
                self.logger.debug("Found existing trained model.")
                return True, None
            # Intermediate checkpoint found
            self.logger.debug('Found checkpoint for epoch: %d' % epoch)
            return False, epoch + 1
        return False, 0

    def _set_data_loaders(self, mode_ids={'train': 0, 'val': 1, 'test': 2}):
        """Re-implement for a train_tester."""
        self._data_loaders = {}

    def _compute_train_loss(self, batch):
        """Compute train loss."""
        accum_loss, accum_losses = zip(*[
            self._compute_loss(batch, step)
            for step in range(len(batch['filenames']))
        ])
        losses = defaultdict(float)
        num_data = sum(len(loss) for loss in accum_loss)
        for loss_dict in accum_losses:
            for key, val in loss_dict.items():
                losses[key] += val.sum().item()
        return (
            sum(torch.sum(loss) for loss in accum_loss) / num_data,
            {key: val / num_data for key, val in losses.items()}
        )

    @torch.no_grad()
    def _compute_validation_loss(self):
        """Compute validation loss."""
        self.training_mode = False
        self.net.eval()
        self.data_loader = self._data_loaders['val']
        loss = 0
        num_data = 0
        print_loss = defaultdict(int)
        for batch in self.data_loader:
            for step in range(len(batch['filenames'])):
                loss_vals, loss_dict = self._compute_loss(batch, step)
                loss += torch.sum(loss_vals).item()
                num_data += len(loss_vals)
                for key, val in loss_dict.items():
                    print_loss[key] += val.sum().item()
        loss /= num_data
        for key in print_loss:
            print_loss[key] /= num_data
        self.net.train()
        self.data_loader = self._data_loaders['train']
        self.training_mode = True
        return loss, print_loss

    @staticmethod
    def _compute_loss(batch, step):
        """
        Compute loss for current batch.

        Re-implement for your problem/net.
        """
        return 0, None

    def _net_forward(self, batch, step):
        """
        Support forward pass of this network.

        Runs a specific network's forward, re-implement for your net.
        """
        return self.net()

    def _net_outputs(self, batch, step):
        """
        Get network outputs for current batch.

        Re-implement for your problem/net.
        """
        return self._net_forward(batch, step)

    def _load_model(self, model_path_name, restore={'optimizer', 'scheduler'}):
        """Load a checkpoint, possibly referring to specific epoch."""
        checkpoint = torch.load(model_path_name, map_location=self._device)
        self.net.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.net.to(self._device)
        trainable = not checkpoint['finished_training']
        if trainable and 'optimizer' in restore:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if trainable and self.scheduler is not None and 'scheduler' in restore:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        return checkpoint['epoch'], checkpoint['finished_training']

    def _save_model(self, model_path_name, epoch, finished_training):
        """Save a checkpoint, possibly referring to specific epoch."""
        checkpoint = {
            'model_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch,
            'finished_training': finished_training
        }
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        torch.save(checkpoint, model_path_name)

    def _setup_loss_criterion(self):
        weights = None
        if self._use_weighted_ce:
            weights = self.annotation_loader.get_class_counts(
                'relations' if 'obj' not in self._task else 'objects'
            )
            weights = 1 / (weights + 1)
            weights = weights / weights.min()
            if len(weights) < self._num_classes:
                weights = np.append(weights, 0)
            weights = torch.from_numpy(weights).float()
        # Set loss
        if self._use_multi_tasking and not self._task.startswith('obj'):
            _criterion = torch.nn.CrossEntropyLoss(
                ignore_index=self._num_classes - 1, reduction='none',
                weight=weights)

            def _criterion_func(scores, targets):
                bg_id = self._num_classes - 1
                scale = 1
                if len(targets[targets == bg_id]) >= 1:
                    scale = len(targets) / len(targets[targets == bg_id])
                return scale * _criterion(scores, targets)
            criterion = _criterion_func  # anonymous func to scale loss
        else:
            criterion = torch.nn.CrossEntropyLoss(
                reduction='none', weight=weights)
        return criterion

    def _setup_optimizer(self):
        logit_params = [  # extra parameters, train with config lr
            param for name, param in self.net.named_parameters()
            if 'top_net' not in name and param.requires_grad]
        backbone_params = [  # backbone net parameters, train with lower lr
            param for name, param in self.net.named_parameters()
            if 'top_net' in name and param.requires_grad]
        return torch.optim.Adam(
            [
                {'params': backbone_params, 'lr': 0.1 * self._learning_rate},
                {'params': logit_params}
            ], lr=self._learning_rate, weight_decay=self._weight_decay)

    def _setup_scheduler(self, optimizer):
        if self._use_early_stopping:
            return EarlyStopping(optimizer, 0.3, patience=self._patience)
        return MultiStepLR(optimizer, [4, 8], gamma=0.3)
