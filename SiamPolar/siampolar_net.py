from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import torch.nn as nn
import torch.nn.functional as F

from pysot.core.config import cfg
from pysot.models.backbone import get_backbone
from pysot.models.neck import get_neck

from pysot.models.fusion import get_feature_fusion
from pysot.models.head import get_mask_head
from pysot.core.xcorr import xcorr_depthwise

# SiamPolar
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

        # build feature fusion
        # self.feature_fusion = get_feature_fusion(cfg.FUSION.TYPE,
        #                                          **cfg.FUSION.KWARGS)
        self.feature_fusion = xcorr_depthwise

        # PolarMask head
        self.mask_head = get_mask_head(cfg.MASK.TYPE,
                                     **cfg.MASK.KWARGS)

    def template(self, z):
        zf = self.backbone(z)
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
        self.zf = zf

    def track(self, x):

        xf = self.backbone(x)
        if cfg.ADJUST.ADJUST:
            xf = self.neck(xf)

        # fusion feature both template and search
        fusion_feature = self.feature_fusion(self.zf, xf)

        cls, loc, cen, mask = self.mask_head(fusion_feature)

        return {
                'cls': cls,
                'cen': cen,
                'loc': loc,
                'mask': mask
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
        label_cls = data['label_cls'].cuda() # batch * 1 * 25 * 25
        # [x1, y1, x2, y2]
        label_loc = data['bbox'].cuda() # batchsize * 4
        # relatedto 36 points location
        label_mask = data['mask'].cuda() # batchsize * 36 * 2

        label_mass_center = data['mass_center'].cuda()
        # feature extractor
        zf = self.backbone(template)
        xf = self.backbone(search)

        # adjust features channels num
        if cfg.ADJUST.ADJUST:
            zf = self.neck(zf)
            xf = self.neck(xf)

        # fusion template and search feature
        fusion_feature = self.feature_fusion(xf, zf)

        # polar mask head
        cls, loc, cen, mask = self.mask_head(fusion_feature)

        cls_loss, loc_loss, cen_loss, mask_loss = self.mask_head.loss(cls,
                                                                      loc,
                                                                      cen,
                                                                      mask,
                                                                      label_loc,
                                                                      label_cls,
                                                                      label_mask,
                                                                      label_mass_center
                                                                      )

        outputs = {}
        outputs['total_loss'] = cfg.TRAIN.CLS_WEIGHT * cls_loss + \
            cfg.TRAIN.LOC_WEIGHT * loc_loss + cfg.TRAIN.CEN_WEIGHT * cen_loss + \
            cfg.TRAIN.MASK_WEIGHT * mask_loss
        outputs['cls_loss'] = cls_loss
        outputs['loc_loss'] = loc_loss
        outputs['cen_loss'] = cen_loss
        outputs['mask_loss'] = mask_loss
        return outputs


