#--------------------------------------------------
#Copyright (c) 
#Licensed under the MIT License
#Written by yeyi (18120438@bjtu.edu.cn)
#--------------------------------------------------

from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F
import time

from pysot.utils.conv2d_util import conv2d_psvf, conv2d_svf, testCon2dPSvf, testCon2dSvf


class Multixcorr_depthwise(nn.Module):

    def __init__(self, channels=3):
        super(Multixcorr_depthwise, self).__init__()

        # addition operator
        self.weight = nn.Parameter(torch.ones(channels))

    def forward(self, x, kernel):
        """
            multi depthwise correlation
            :param x:
            :param kernel:
            :return:
            """
        batch = kernel.size(0)
        batch = kernel.size(0)
        channel = kernel.size(1)
        x = x.view(1, batch * channel, x.size(2), x.size(3))
        kernel = kernel.view(batch * channel, 1, kernel.size(2), kernel.size(3))
        # Square variance formula depthwise correlation x^2 - y^2
        # out_1 = conv2d_svf(x, kernel, groups=batch * channel)
        out_1 = testCon2dSvf(x, kernel, groups=batch * channel)
        out_1 = out_1.view(batch, channel, out_1.size(2), out_1.size(3))
        # Perfect Square variance formula depthwise correlation (x - y)^2
        # out_2 = conv2d_psvf(x, kernel, groups=batch * channel)
        out_2 = testCon2dPSvf(x, kernel, groups=batch * channel)
        out_2 = out_2.view(batch, channel, out_2.size(2), out_2.size(3))
        # normal depthwise correlation x * y
        out_3 = F.conv2d(x, kernel, groups=batch * channel)
        out_3 = out_3.view(batch, channel, out_3.size(2), out_3.size(3))

        # addition 3 channels feature
        weight = F.softmax(self.weight, 0)

        out = weight[0] * out_1 + weight[1] * out_2 + weight[2] * out_3

        return out
