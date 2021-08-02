# -*- coding: utf-8 -*-
"""A baseline using only visual features."""

from common.models.sg_generator import HGATNet
from research.src.train_testers import SGGTrainTester


def train_test(config, obj_classifier=None, teacher=None):
    """Train and test a net."""
    net = HGATNet(config)
    train_tester = SGGTrainTester(
        net, config, {'images'}, obj_classifier, teacher
    )
    train_tester.train_test()
