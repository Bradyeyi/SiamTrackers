#--------------------------------------------------
#Copyright (c)
#Licensed under the MIT License
#Written by yeyi (18120438@bjtu.edu.cn)
#--------------------------------------------------

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

from pysot.models.fusion.multixcorr_depthwise import Multixcorr_depthwise
from pysot.models.fusion.dwconv import KernelDWConv2d
from pysot.models.fusion.feature_combination import Feature_Combination

FUSIONS = {
         'Feature_Combination': Feature_Combination,
         'Multixcorr_depthwise': Multixcorr_depthwise,
         'KernelDWConv2d': KernelDWConv2d,
        }

def get_feature_fusion(name, **kwargs):
    return FUSIONS[name](**kwargs)
