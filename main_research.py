# -*- coding: utf-8 -*-
"""Model training/testing pipeline."""

import argparse
import json
import os

import yaml

from common.models import load_model
from research.research_config import ResearchConfig
from research.src.models.object_classifier import (
    load_classifier,
    object_classifier
)
from research.src.models.object_detector import object_detector
from research.src.models import sg_generator
from research.src.models import sg_projector
from research.src.models import grounder
from research.src.models import transformer

MODELS = {
    'phrase_transformer': transformer.phrase_transformer,
    'phrase_crosstransformer': transformer.phrase_crosstransformer,
    'phrase_decoder': transformer.phrase_decoder,
    'multitask_decoder': transformer.multitask_decoder,
    'atr_net': sg_generator.atr_net,
    'ground_bbox_net': grounder.ground_bbox_net,
    'independent_net': sg_generator.independent_net,
    'language_net': sg_generator.language_net,
    'language_projector': sg_projector.language_projector,
    'lang_spat_net': sg_generator.lang_spat_net,
    'visual_lang_spat_net': sg_generator.visual_lang_spat_net,
    # 'lang_spat_moe_net': lang_spat_moe_net,
    # 'lang_spat_cosim_net': lang_spat_cosim_net,
    # 'dynamic_lang_net': dynamic_lang_net,
    # 'lang_spat_selfdistil_net': lang_spat_selfdistil_net,
    'seq_att_net': grounder.seq_att_net,
    'lang_spat_projector': sg_projector.lang_spat_projector,
    'motifs_net': sg_generator.motifs_net,
    'gps_net': sg_generator.gps_net,
    'hgat_net': sg_generator.hgat_net,
    'object_classifier': object_classifier,
    'object_detector': object_detector,
    'ref_rel_conditioned_net': grounder.ref_rel_conditioned_net,
    'ref_rel_parallel_net': grounder.ref_rel_parallel_net,
    'parsing_net': grounder.parsing_net,
    'parsing_dyn_lang_net': grounder.parsing_dyn_lang_net,
    'ref_rel_combinator_net': grounder.ref_rel_combinator_net,
    'ref_rel_dynamic_conv_net': grounder.ref_rel_dynamic_conv_net,
    'reldn_net': sg_generator.reldn_net,
    'spatial_net': sg_generator.spatial_net,
    'spatial_projector': sg_projector.spatial_projector,
    'uvtranse_net': sg_generator.uvtranse_net,
    'visual_net': sg_generator.visual_net,
    'vtranse_net': sg_generator.vtranse_net,
    'visual_projector': sg_projector.visual_projector,
    'visual_spat_attention_net': sg_generator.visual_spat_attention_net,
    'visual_spat_attention_projector': sg_projector.visual_spat_attention_projector,
    'visual_spat_net': sg_generator.visual_spat_net,
    'visual_spat_projector': sg_projector.visual_spat_projector,
    'spatlang_negatives_ranker_net': sg_generator.spatlang_negatives_ranker_net
}

with open('prerequisites_config.yaml', 'r') as fid:
    CONFIG = yaml.load(fid, Loader=yaml.FullLoader)


class ParseKwargs(argparse.Action):
    def __call__(self, parser, namespace, values, option_string=None):
        setattr(namespace, self.dest, dict())
        for value in values:
            key, value = value.split('=')
            if value in ['True', 'False']:
                value = value == 'True'
            else:
                try:
                    value = float(value)
                except ValueError:
                    pass
            getattr(namespace, self.dest)[key] = value


def parse_args():
    """Parse input arguments."""
    parser = argparse.ArgumentParser()
    # Model to train/test and peculiar parameters
    parser.add_argument(
        '--model', dest='model', help='Model to train (see main.py)',
        type=str, default='lang_spat_net'
    )
    parser.add_argument(
        '--misc_params', nargs='*', action=ParseKwargs, default=None
    )
    parser.add_argument(
        '--object_classifier', dest='object_classifier',
        help='Name of classifier model to use if task is sgcls',
        type=str, default='object_classifier'
    )
    parser.add_argument(
        '--teacher', dest='teacher',
        help='Name of teacher model to use for distillation',
        type=str, default=None
    )
    parser.add_argument(
        '--teacher_name', dest='teacher_name',
        help='Name of teacher net (e.g. visual_net2_predcls_VRD)',
        type=str, default=None
    )
    # Dataset/task parameters
    parser.add_argument(
        '--dataset', dest='dataset', help='Dataset codename (e.g. VG200)',
        type=str, default='VRD'
    )
    parser.add_argument(
        '--task', dest='task',
        help='Task to solve, check config.py for supported tasks',
        type=str, default='preddet'
    )
    parser.add_argument(
        '--net_name', dest='net_name', help='Name of trained model',
        type=str, default=''
    )
    parser.add_argument(
        '--phrase_recall', dest='phrase_recall',
        help='Whether to evaluate phrase recall',
        action='store_true'
    )
    parser.add_argument(
        '--test_dataset', dest='test_dataset',
        help='Dataset to evaluate on, if different than train dataset',
        type=str, default=None
    )
    # Specific task parameters: data handling
    parser.add_argument(
        '--annotations_per_batch', dest='annotations_per_batch',
        help='Batch size in terms of annotations (e.g. relationships)',
        type=int, default=128
    )
    parser.add_argument(
        '--not_augment_annotations', dest='not_augment_annotations',
        help='Do not augment annotations with box/image distortion',
        action='store_true'
    )
    parser.add_argument(
        '--bg_perc', dest='bg_perc',
        help='Percentage of background annotations',
        type=float, default=None
    )
    parser.add_argument(
        '--filter_duplicate_rels', dest='filter_duplicate_rels',
        help='Whether to filter relations annotated more than once',
        action='store_true'
    )
    parser.add_argument(
        '--filter_multiple_preds', dest='filter_multiple_preds',
        help='Whether to sample a single predicate per object pair',
        action='store_true'
    )
    parser.add_argument(
        '--max_train_samples', dest='max_train_samples',
        help='Keep classes at most such many training samples',
        type=int, default=None
    )
    parser.add_argument(
        '--num_tail_classes', dest='num_tail_classes',
        help='Keep such many classes with the fewest training samples',
        type=int, default=None
    )
    parser.add_argument(
        '--use_negative_samples', dest='use_negative_samples',
        help='Whether to use extra annotations from negative samples',
        action='store_true'
    )
    # Evaluation parameters
    parser.add_argument(
        '--compute_accuracy', dest='compute_accuracy',
        help='For preddet only, measure accuracy instead of recall',
        action='store_true'
    )
    parser.add_argument(
        '--use_merged', dest='use_merged',
        help='Evaluate with merged predicate annotations',
        action='store_true'
    )
    # General model parameters
    parser.add_argument(
        '--is_not_context_projector', dest='is_not_context_projector',
        help='Do not treat this projector as a context projector',
        action='store_true'
    )
    parser.add_argument(
        '--is_cos_sim_projector', dest='is_cos_sim_projector',
        help='Maximize cos. similarity between features and learned weights',
        action='store_true'
    )
    # Specific task parameters: loss function
    parser.add_argument(
        '--not_use_multi_tasking', dest='not_use_multi_tasking',
        help='Do not use multi-tasking to detect "no interaction" cases',
        action='store_true'
    )
    parser.add_argument(
        '--use_weighted_ce', dest='use_weighted_ce',
        help='Use weighted cross-entropy',
        action='store_true'
    )
    # Training parameters
    parser.add_argument(
        '--batch_size', dest='batch_size',
        help='Batch size in terms of images',
        type=int, default=None
    )
    parser.add_argument(
        '--epochs', dest='epochs', help='Number of training epochs',
        type=int, default=None
    )
    parser.add_argument(
        '--learning_rate', dest='learning_rate',
        help='Learning rate of classification layers (not backbone)',
        type=float, default=0.002
    )
    parser.add_argument(
        '--weight_decay', dest='weight_decay',
        help='Weight decay of optimizer',
        type=float, default=None
    )
    # Learning rate policy
    parser.add_argument(
        '--apply_dynamic_lr', dest='apply_dynamic_lr',
        help='Adapt learning rate so that lr / batch size = const',
        action='store_true'
    )
    parser.add_argument(
        '--not_use_early_stopping', dest='not_use_early_stopping',
        help='Do not use early stopping learning rate policy',
        action='store_true'
    )
    parser.add_argument(
        '--not_restore_on_plateau', dest='not_restore_on_plateau',
        help='Do not restore best model on validation plateau',
        action='store_true'
    )
    parser.add_argument(
        '--patience', dest='patience',
        help='Number of epochs to consider a validation plateu',
        type=int, default=1
    )
    # Other data loader parameters
    parser.add_argument(
        '--commit', dest='commit',
        help='Commit name to tag model',
        type=str, default=''
    )
    parser.add_argument(
        '--num_workers', dest='num_workers',
        help='Number of workers employed by data loader',
        type=int, default=2
    )
    parser.add_argument(
        '--rel_batch_size', dest='rel_batch_size',
        help='Number of relations per sub-batch (memory issues)',
        type=int, default=128
    )
    parser.add_argument(
        '--negative_loss', dest='negative_loss',
        help='Type of negative loss to use, see _negatives_loss()',
        type=str, default=None
    )
    parser.add_argument(
        '--neg_classes', dest='neg_classes', nargs='+',
        help='Classes to implement negative loss, all if not set',
        type=int, default=None
    )
    parser.add_argument(
        '--use_graphl_loss', dest='use_graphl_loss',
        help='Whether to use graphical contrastive losses (Zhang 19)',
        action='store_true'
    )
    parser.add_argument(
        '--use_consistency_loss', dest='use_consistency_loss',
        help='Whether to use consistency loss',
        action='store_true'
    )
    parser.add_argument(
        '--test_on_negatives', dest='test_on_negatives',
        help='Whether to test on negative labels',
        action='store_true'
    )
    parser.add_argument(
        '--overfit', dest='overfit',
        help='Whether to train only on a handful of samples to overfit',
        action='store_true'
    )
    return parser.parse_args()


def main():
    """Train and test a network pipeline."""
    args = parse_args()
    model = MODELS[args.model]
    if os.path.exists('/gpu-data/mdiom/'):
        _path = '/gpu-data/mdiom/'
    elif os.path.exists('/gpu-data2/ngan/'):
        _path = '/gpu-data2/ngan/'
    elif os.path.exists('/gpu-data/ngan/'):
        _path = '/gpu-data/ngan/'
    else:
        _path = CONFIG['prerequisites_path']
    cfg = ResearchConfig(net_name=args.net_name if args.net_name else args.model, phrase_recall=args.phrase_recall,
                         test_dataset=args.test_dataset, annotations_per_batch=args.annotations_per_batch,
                         augment_annotations=not args.not_augment_annotations, compute_accuracy=args.compute_accuracy,
                         use_merged=args.use_merged, use_multi_tasking=not args.not_use_multi_tasking,
                         use_weighted_ce=args.use_weighted_ce, batch_size=args.batch_size, epochs=args.epochs,
                         learning_rate=args.learning_rate, weight_decay=args.weight_decay,
                         apply_dynamic_lr=args.apply_dynamic_lr, use_early_stopping=not args.not_use_early_stopping,
                         restore_on_plateau=not args.not_restore_on_plateau, patience=args.patience, commit=args.commit,
                         num_workers=args.num_workers, use_consistency_loss=args.use_consistency_loss,
                         use_graphl_loss=args.use_graphl_loss,
                         misc_params=args.misc_params, dataset=args.dataset, task=args.task, bg_perc=args.bg_perc,
                         filter_duplicate_rels=args.filter_duplicate_rels,
                         filter_multiple_preds=args.filter_multiple_preds, max_train_samples=args.max_train_samples,
                         num_tail_classes=args.num_tail_classes, use_negative_samples=args.use_negative_samples,
                         rel_batch_size=args.rel_batch_size, negative_loss=args.negative_loss,
                         neg_classes=args.neg_classes, is_context_projector=not args.is_not_context_projector,
                         is_cos_sim_projector=args.is_cos_sim_projector, prerequisites_path=_path,
                         test_on_negatives=args.test_on_negatives,
                         overfit=args.overfit)
    obj_classifier = None
    teacher = None
    if args.task == 'sgcls':
        obj_classifier = load_classifier(args.object_classifier, cfg)
    if args.teacher is not None:
        teacher_name = '_'.join([args.teacher] + cfg.net_name.split('_')[-2:])
        if args.teacher_name is not None:
            teacher_name = args.teacher_name
        teacher = load_model(
            cfg, args.teacher, teacher_name,
            path=cfg.prerequisites_path + 'models/' + teacher_name + '/')
        for param in teacher.parameters():
            param.requires_grad = False
    model.train_test(cfg, obj_classifier, teacher)


if __name__ == "__main__":
    main()
