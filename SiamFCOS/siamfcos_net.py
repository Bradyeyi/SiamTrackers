from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.backbone import get_backbone
from pysot.models.head import get_fcos_head
from pysot.models.neck import get_neck
from pysot.core.xcorr import xcorr_depthwise
from SiamFCOS.utils_fcos.cen_utils import compute_locations
from SiamFCOS.utils_fcos.loss_fcos import make_fcos_loss_evaluator

# SiamFCOS Model Neural network architecture
class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()

        # build backbone
        self.backbone = get_backbone(cfg.BACKBONE.TYPE,
                                     **cfg.BACKBONE.KWARGS)

        # build adjust layer
        if cfg.ADJUST.ADJUST:
            self.neck = get_neck(cfg.ADJUST.TYPE,
                                 **cfg.ADJUST.KWARGS)

        # build fcos head
        self.fcos_head = get_fcos_head(cfg.FCOS.TYPE,
                                     **cfg.FCOS.KWARGS)

        # build response map
        self.xcorr_depthwise = xcorr_depthwise

        # build loss
        self.loss_evaluator = make_fcos_loss_evaluator(cfg)

        # build activate function, 进行特征融合
        # self.down = nn.ConvTranspose2d(256 * 3, 256, 1, 1)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):

        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        if cfg.FCOS.TYPE == 'CARHead':
            feature = xcorr_depthwise(xf, self.zf)
            cls, cen, loc = self.fcos_head(feature)
        else:
            cls, cen, loc = self.fcos_head(self.zf, xf)

        return {
                'cls': cls,
                'cen': cen,
                'loc': loc,
                # 'mask': mask if cfg.MASK.MASK else None
               }


    def log_softmax(self, cls):
        b, a2, h, w = cls.size()
        # print(a2)
        cls = cls.view(b, 2, a2//2, h, w)
        cls = cls.permute(0, 2, 3, 4, 1).contiguous()
        cls = F.log_softmax(cls, dim=4)
        return cls

    def forward(self, data):
        """ only used in training
        """
        template = data['template'].cuda()
        search = data['search'].cuda()
        label_cls = data['label_cls'].cuda() # 64 * 1 * 25 * 25
        label_loc = data['bbox'].cuda() # 64 * 4

        # print(label_cls)
        # get feature
        zf = self.backbone(template)
        xf = self.backbone(search)

        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        if cfg.FCOS.TYPE == 'CARHead':
            feature = xcorr_depthwise(xf, zf)
            cls, cen, loc = self.fcos_head(feature)
        else:
            cls, cen, loc = self.fcos_head(zf, xf)

        locations = compute_locations(cls, 8)
        # print(locations)
        # # locations: 625 * 2
        # print(locations.size())
        #
        if cfg.FCOS.TYPE == 'CARHead':
            cls = self.log_softmax(cls) #batchsize * 1 * 25 * 25 * 2

        cls_loss, loc_loss,  cen_loss = self.loss_evaluator(
            locations,
            cls,
            loc,
            cen, label_cls, label_loc
        )

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        return outputs


