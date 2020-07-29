#--------------------------------------------------
#Copyright (c) 
#Licensed under the MIT License
#Written by yeyi (18120438@bjtu.edu.cn)
#--------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F

from .dwconv import KernelDWConv2d
from .multixcorr_depthwise import Multixcorr_depthwise
from pysot.core.xcorr import xcorr_depthwise

class Feature_Combination(nn.Module):
    """
    define Feature_Combination Module : refer Ocean tracker
    """    
    def __init__(self, in_channels, out_channels):
        super(Feature_Combination, self).__init__()

        # same size dilation = (1, 1)
        self.matrix11_t = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix11_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # same size dilation = (2, 1) : H / 2 , W
        self.matrix12_t = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix12_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, dilation=(2, 1)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # same size dilation = (1, 2) : H, W / 2
        self.matrix21_t = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        self.matrix21_s = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, bias=False, dilation=(1, 2)),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

        # cross correlation
        self.feature_fusion = xcorr_depthwise

        # addition operator
        self.weight = nn.Parameter(torch.ones(3))
        
        
    def forword(self, search, template):

        # Convolution with different dilation
        template_11 = self.matrix11_t(template)
        search_11 = self.matrix11_s(search)

        template_12 = self.matrix12_t(template)
        search_12 = self.matrix12_s(search)

        template_21 = self.matrix21_t(template)
        search_21 = self.matrix21_s(search)

        # cross correlation
        fusion_11 = self.feature_fusion(search_11, template_11)
        fusion_12 = self.feature_fusion(search_12, template_12)
        fusion_21 = self.feature_fusion(search_21, template_21)

        # weight fusion
        weight = F.softmax(self.weight, 0)
        fusion = weight[0] * fusion_11 + weight[1] * fusion_12 + weight[2] * fusion_21

        return fusion
        