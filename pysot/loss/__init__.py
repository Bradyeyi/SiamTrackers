# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.loss.loss import select_cross_entropy_loss, get_cls_loss, weight_l1_loss
from pysot.loss.focal_loss import FocalLoss
from pysot.loss.iou_loss import IOULoss

LOSSES = {
    'select_cross_entropy_loss':select_cross_entropy_loss,
    'get_cls_loss':get_cls_loss,
    'weight_l1_loss':weight_l1_loss,
    'focal_loss': FocalLoss,
    'iou_loss': IOULoss
}

def build_loss(name, **kwargs):
    return LOSSES[name](**kwargs)
