# Copyright (c) SenseTime. All Rights Reserved.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.head.mask import MaskCorr, Refine
from pysot.models.head.rpn import UPChannelRPN, DepthwiseRPN, MultiDepthwiseRPN, KernelDWConv2dRPN, MultiRPN
from pysot.models.head.fcos import DepthwiseFCOS, CARHead, MultiFCOS
from pysot.models.head.polarmask_head import PolarMaskHead

RPNS = {
        'UPChannelRPN': UPChannelRPN,
        'DepthwiseRPN': DepthwiseRPN,
        'MultiRPN': MultiRPN,
        'MultiDepthwiseRPN': MultiDepthwiseRPN,
        'KernelDWConv2dRPN': KernelDWConv2dRPN
       }

FCOS = {
        'DepthwiseFCOS': DepthwiseFCOS,
        'CARHead': CARHead,
        'MultiFCOS': MultiFCOS,
}

MASKS = {
         'MaskCorr': MaskCorr,
         'PolarMaskHead': PolarMaskHead,
        }

REFINE = {
          'Refine': Refine,
         }


def get_rpn_head(name, **kwargs):
    return RPNS[name](**kwargs)


def get_mask_head(name, **kwargs):
    return MASKS[name](**kwargs)


def get_refine_head(name):
    return REFINE[name]()

def get_fcos_head(name, **kwargs):
    return FCOS[name](**kwargs)
