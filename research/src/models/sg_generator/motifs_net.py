# -*- coding: utf-8 -*-
"""Neural Motifs Network by Zellers et al., 2018."""

from common.models.sg_generator import MotifsNet
from research.src.train_testers import SGGTrainTester


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a net."""
    net = MotifsNet(config)
    train_tester = SGGTrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()
