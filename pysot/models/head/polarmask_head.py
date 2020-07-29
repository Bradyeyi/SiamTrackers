#--------------------------------------------------
#Copyright (c)
#Licensed under the MIT License
#Written by yeyi (18120438@bjtu.edu.cn)
#--------------------------------------------------

import torch
import torch.nn as nn
import math

from pysot.core.config import cfg
from pysot.loss import build_loss
# from pysot.models.dcn import ModulatedDeformConvPack
from modules import ModulatedDeformConvPack
from pysot.loss.focal_loss import FocalLoss
from pysot.loss.iou_loss import IOULoss
from pysot.loss.mask_iou_loss import MaskIOULoss

INF = 1e8

class PolarMaskHead(torch.nn.Module):
    def __init__(self, in_channels,
                 stacked_convs=4,
                 stride= 8,
                 align_head = False,
                 use_dcn=False,
                 ):
        """
        Arguments:
            in_channels (int): number of channels of the input feature
        """
        super(PolarMaskHead, self).__init__()
        # Conv layer nums
        self.stacked_convs = stacked_convs

        # data deal
        self.stride = stride
        self.use_dcn = use_dcn
        self.align_head = align_head

        # build loss
        self.cls_loss = FocalLoss(
            cfg.FCOS.FOCAL_LOSS_ALPHA,
            cfg.FCOS.FOCAL_LOSS_GAMMA,
            reduction="sum"
        )
        self.iou_loss_type = cfg.MASK.IOU_LOSS_TYPE
        self.bbox_loss = IOULoss(self.iou_loss_type)
        self.cen_loss = nn.BCEWithLogitsLoss()
        self.mask_loss = MaskIOULoss()

        # dataset deal
        self.radius = 1.5
        self.center_sample = True
        self.use_mass_center = False

        self.regress_range = (-1, 64)

        # Moule initialize
        num_classes = 1
        cls_tower = []
        bbox_tower = []
        mask_tower = []
        for i in range(stacked_convs):
            if not self.use_dcn:
                cls_tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                cls_tower.append(nn.GroupNorm(32, in_channels))
                cls_tower.append(nn.ReLU(inplace=True))
                bbox_tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                bbox_tower.append(nn.GroupNorm(32, in_channels))
                bbox_tower.append(nn.ReLU(inplace=True))
                mask_tower.append(
                    nn.Conv2d(
                        in_channels,
                        in_channels,
                        kernel_size=3,
                        stride=1,
                        padding=1
                    )
                )
                mask_tower.append(nn.GroupNorm(32, in_channels))
                mask_tower.append(nn.ReLU(inplace=True))
            else:
                cls_tower.append(
                    ModulatedDeformConvPack(
                        in_channels,
                        in_channels,
                        3,#kernel_size
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    ))
                cls_tower.append(nn.GroupNorm(32, in_channels))
                cls_tower.append(nn.ReLU(inplace=True))
                bbox_tower.append(
                    ModulatedDeformConvPack(
                        in_channels,
                        in_channels,
                        3,# kernel size
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    )
                )
                bbox_tower.append(nn.GroupNorm(32, in_channels))
                bbox_tower.append(nn.ReLU(inplace=True))
                mask_tower.append(
                    ModulatedDeformConvPack(
                        in_channels,
                        in_channels,
                        3,  # kernel size
                        stride=1,
                        padding=1,
                        dilation=1,
                        deformable_groups=1,
                    )
                )
                mask_tower.append(nn.GroupNorm(32, in_channels))
                mask_tower.append(nn.ReLU(inplace=True))

        self.add_module('cls_tower', nn.Sequential(*cls_tower))
        self.add_module('bbox_tower', nn.Sequential(*bbox_tower))
        self.add_module('mask_tower',nn.Sequential(*mask_tower))

        self.cls_logits = nn.Conv2d(
            in_channels, num_classes, kernel_size=3, stride=1,
            padding=1
        )
        self.bbox_pred = nn.Conv2d(
            in_channels, 4, kernel_size=3, stride=1,
            padding=1
        )
        self.centerness = nn.Conv2d(
            in_channels, 1, kernel_size=3, stride=1,
            padding=1
        )
        self.mask_pred = nn.Conv2d(
            in_channels, 36, kernel_size=3, stride=1,
            padding=1
        )

        # initialization layers
        # if not use_dcn
        if not self.use_dcn:
            for modules in [self.cls_tower, self.bbox_tower,
                            self.mask_tower,self.cls_logits,
                            self.bbox_pred,self.centerness,
                            self.mask_pred]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)
        else:
            for modules in [self.cls_logits, self.bbox_pred,
                            self.centerness, self.mask_pred]:
                for l in modules.modules():
                    if isinstance(l, nn.Conv2d):
                        torch.nn.init.normal_(l.weight, std=0.01)
                        torch.nn.init.constant_(l.bias, 0)

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):

        logits = self.cls_logits(self.cls_tower(x))
        bbox_tower = self.bbox_tower(x)
        bbox_reg = torch.exp(self.bbox_pred(bbox_tower))
        centerness = self.centerness(bbox_tower)
        mask_pred = torch.exp(self.mask_pred(self.mask_tower(x)))
        return logits, bbox_reg, centerness, mask_pred

    def get_point(self, featmap_size, stride, dtype, device):
        # cls size h * w
        h, w = featmap_size
        x_range = torch.arange(
            0, w * stride, stride, dtype=dtype, device=device)
        y_range = torch.arange(
            0, h * stride, stride, dtype=dtype, device=device)
        # generate location of images
        y, x = torch.meshgrid(y_range, x_range)
        # refer to SiamCAR
        point = torch.stack(
            (x.reshape(-1), y.reshape(-1)), dim=-1) + 32
        return point

    def polar_target(self, points, gt_labels, gt_bboxs, gt_masks, mass_center):

        num_points = points.size(0)
        # points: 625 * 2
        xs, ys = points[:, 0], points[:, 1]

        bboxes = gt_bboxs # 64 * 4

        areas = (bboxes[:,2] - bboxes[:,0] + 1) * (bboxes[:,3] - bboxes[:,1] + 1)
        areas = areas[None].repeat(num_points, 1)

        labels = gt_labels.view(625, -1)

        # 计算location坐标和真实边框之间的距离
        l = xs[:, None] - bboxes[:, 0][None].float()
        t = ys[:, None] - bboxes[:, 1][None].float()
        r = bboxes[:, 2][None].float() - xs[:, None]
        b = bboxes[:, 3][None].float() - ys[:, None]
        reg_targets = torch.stack([l, t, r, b], dim=2)

        # mass center : get from siampolar_dataset
        mass_centers = mass_center[None].expand(num_points, gt_masks.size(0), 2)
        #center sample
        if self.center_sample:
            if self.use_mass_center:
                inside_gt_bbox_mask = self.get_mask_sample_region(bboxes,
                                                                  mass_centers,
                                                                  self.stride,
                                                                  num_points,
                                                                  xs,
                                                                  ys,
                                                                  radius=self.radius)
            else:
                # box center sample
                inside_gt_bbox_mask = self.get_sample_region(bboxes,
                                                             self.stride,
                                                             num_points,
                                                             xs,
                                                             ys,
                                                             radius=self.radius)

        else:
            # no center sample, use all points in gt bbox
            inside_gt_bbox_mask = reg_targets.min(dim=2)[0] > 0

        # condition2: limit the regression range for each location
        max_reg_targets = reg_targets.max(dim=2)[0]

        # limit the regression range for each location
        is_cared_in_the_level = \
            (max_reg_targets >= -1) & \
            (max_reg_targets <= 64)

        inside_gt_bbox = inside_gt_bbox_mask & is_cared_in_the_level
        areas[inside_gt_bbox == 1] = INF

        labels[areas==INF] = 1
        # get mask targets 625 2   64 * 36 * 2 ==>> 64 625 36
        mask_targets = torch.zeros(len(bboxes), num_points, 36).float().cuda()

        for i in range(bboxes.size(0)):
            gt_mask = gt_masks[i,...]
            d_x = xs[:, None] - gt_mask[:, 0][None].float()
            d_y = ys[:, None] - gt_mask[:, 1][None].float()
            distance = torch.sqrt(d_x**2 + d_y**2)
            mask_targets[i, :, :] = distance

        return labels.permute(1,0).contiguous(), reg_targets.permute(1,0,2).contiguous(), \
               mask_targets

    def get_mask_sample_region(self, gt_bb, mask_center, stride, num_points, gt_xs, gt_ys, radius=1):
        center_y = mask_center[..., 0]
        center_x = mask_center[..., 1]
        center_gt = gt_bb.new_zeros(gt_bb.shape)
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)

        beg = 0

        end = beg + num_points
        stride = stride * radius
        xmin = center_x[beg:end] - stride
        ymin = center_y[beg:end] - stride
        xmax = center_x[beg:end] + stride
        ymax = center_y[beg:end] + stride
        # limit sample region in gt
        center_gt[beg:end, :, 0] = torch.where(xmin > gt_bb[beg:end, :, 0], xmin, gt_bb[beg:end, :, 0])
        center_gt[beg:end, :, 1] = torch.where(ymin > gt_bb[beg:end, :, 1], ymin, gt_bb[beg:end, :, 1])
        center_gt[beg:end, :, 2] = torch.where(xmax > gt_bb[beg:end, :, 2], gt_bb[beg:end, :, 2], xmax)
        center_gt[beg:end, :, 3] = torch.where(ymax > gt_bb[beg:end, :, 3], gt_bb[beg:end, :, 3], ymax)
        beg = end

        left = gt_xs - center_gt[..., 0]
        right = center_gt[..., 2] - gt_xs
        top = gt_ys - center_gt[..., 1]
        bottom = center_gt[..., 3] - gt_ys
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0  # 上下左右都>0 就是在bbox里面
        return inside_gt_bbox_mask

    def get_sample_region(self, gt, stride, num_points, gt_xs, gt_ys, radius=1.0):
        '''
        This code is from
        https://github.com/yqyao/FCOS_PLUS/blob/0d20ba34ccc316650d8c30febb2eb40cb6eaae37/
        maskrcnn_benchmark/modeling/rpn/fcos/loss.py#L42
        '''
        num_gts = gt.shape[0] # 64
        K = len(gt_xs) # 625
        gt = gt[None].expand(K, num_gts, 4)
        center_x = (gt[..., 0] + gt[..., 2]) / 2
        center_y = (gt[..., 1] + gt[..., 3]) / 2
        center_gt = gt.new_zeros(gt.shape)
        # no gt
        if center_x[..., 0].sum() == 0:
            return gt_xs.new_zeros(gt_xs.shape, dtype=torch.uint8)
        beg = 0
        end = beg + num_points
        stride = stride * radius
        xmin = center_x[beg:end] - stride
        ymin = center_y[beg:end] - stride
        xmax = center_x[beg:end] + stride
        ymax = center_y[beg:end] + stride
        # limit sample region in gt
        center_gt[beg:end, :, 0] = torch.where(
            xmin > gt[beg:end, :, 0], xmin, gt[beg:end, :, 0]
        )
        center_gt[beg:end, :, 1] = torch.where(
            ymin > gt[beg:end, :, 1], ymin, gt[beg:end, :, 1]
        )
        center_gt[beg:end, :, 2] = torch.where(
            xmax > gt[beg:end, :, 2],
            gt[beg:end, :, 2], xmax
        )
        center_gt[beg:end, :, 3] = torch.where(
            ymax > gt[beg:end, :, 3],
            gt[beg:end, :, 3], ymax
        )

        left = gt_xs[:, None] - center_gt[..., 0].float()
        right = center_gt[..., 2].float() - gt_xs[:, None]
        top = gt_ys[:, None] - center_gt[..., 1].float()
        bottom = center_gt[..., 3].float() - gt_ys[:, None]
        center_bbox = torch.stack((left, top, right, bottom), -1)
        inside_gt_bbox_mask = center_bbox.min(-1)[0] > 0
        return inside_gt_bbox_mask

    def polar_centerness_target(self, pos_mask_targets):
        # only calculate pos centerness targets, otherwise there may be nan
        centerness_targets = (pos_mask_targets.min(dim=-1)[0] / pos_mask_targets.max(dim=-1)[0])
        return torch.sqrt(centerness_targets)

    def loss(self,
             cls_score,
             bbox_pred,
             centerness,
             mask_pred,
             gt_bbox,
             gt_label,
             gt_mask,
             mass_center
             ):
        num_imgs = cls_score.size(0)
        featmap_sizes = cls_score.size()[-2:]
        points = self.get_point(featmap_sizes, self.stride, bbox_pred[0].dtype,
                                           bbox_pred[0].device)

        # deal groundtruth label, attain polarmask related annotation
        labels, bbox_targets, mask_targets = self.polar_target(points, gt_label, gt_bbox, gt_mask, mass_center)

        # flatten cls_score, bbox_pred , centerness, mask_pred
        cls_score_flatten = (cls_score.permute(0, 2, 3, 1).contiguous().view(-1))
        bbox_pred_flatten = (bbox_pred.permute(0, 2, 3, 1).contiguous().view(-1, 4))
        centerness_flatten = (centerness.permute(0, 2, 3, 1).contiguous().view(-1))
        mask_pred_flatten = (mask_pred.permute(0, 2, 3, 1).contiguous().view(-1, 36))

        labels_flatten = labels.view(-1)
        bbox_targets_flatten = bbox_targets.view(-1, 4)
        mask_targets_flatten = mask_targets.view(-1, 36)


        # get the positive object location
        pos_inds = torch.nonzero(labels_flatten > 0).squeeze(1)
        num_pos = len(pos_inds)

        pos_bbox_preds = bbox_pred_flatten[pos_inds]
        pos_centerness = centerness_flatten[pos_inds]
        pos_mask_preds = mask_pred_flatten[pos_inds]

        pos_bbox_targets = bbox_targets_flatten[pos_inds]
        pos_mask_targets = mask_targets_flatten[pos_inds]

        loss_cls = self.cls_loss(
            cls_score_flatten, labels_flatten,
            avg_factor= num_pos + num_imgs)  # avoid num_pos is 0

        if num_pos > 0:

            pos_centerness_targets = self.polar_centerness_target(pos_mask_targets)

            # centerness weighted iou loss
            loss_bbox = self.bbox_loss(
                pos_bbox_preds,
                pos_bbox_targets,
                weight=pos_centerness_targets)

            loss_mask = self.mask_loss(pos_mask_preds,
                                       pos_mask_targets,
                                       weight=pos_centerness_targets,
                                       avg_factor=pos_centerness_targets.sum())

            loss_centerness = self.cen_loss(pos_centerness,
                                            pos_centerness_targets)
        else:
            loss_bbox = pos_bbox_preds.sum()
            loss_mask = pos_mask_preds.sum()
            loss_centerness = pos_centerness.sum()

        return loss_cls, loss_bbox, loss_centerness, loss_mask,

class AlignHead(nn.Module):
    def __init__(self):
        super(AlignHead, self).__init__()
