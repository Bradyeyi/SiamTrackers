#--------------------------------------------------
#Copyright (c) from https://github.com/CoinCheung/pytorch-loss/blob/master/focal_loss.py
#Licensed under the MIT License
#Written by yeyi (18120438@bjtu.edu.cn)
#--------------------------------------------------

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean',
                 avg_factor=None):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.avg_factor = avg_factor
        self.crit = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, label, avg_factor=None):
        '''
        args:
            logits: tensor of shape (N)
            label: tensor of shape(N)
        '''
        # compute loss
        logits = logits.float() # use fp32 if logits is fp16
        with torch.no_grad():
            alpha = torch.empty_like(logits).fill_(1 - self.alpha)
            alpha[label == 1] = self.alpha

        probs = torch.sigmoid(logits)
        pt = torch.where(label == 1, probs, 1 - probs)
        ce_loss = self.crit(logits, label.float())
        loss = (alpha * torch.pow(1 - pt, self.gamma) * ce_loss)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            if avg_factor:
                loss = loss.sum() / avg_factor
            else:
                loss = loss.sum()
        return loss

# version 2: user derived grad computation
class FocalSigmoidLossFuncV2(torch.autograd.Function):
    '''
    compute backward directly for better numeric stability
    '''
    @staticmethod
    def forward(ctx, logits, label, alpha, gamma):
        logits = logits.float()
        coeff = torch.empty_like(logits).fill_(1 - alpha)
        coeff[label == 1] = alpha

        probs = torch.sigmoid(logits)
        log_probs = torch.where(logits >= 0,
                F.softplus(logits, -1, 50),
                logits - F.softplus(logits, 1, 50))
        log_1_probs = torch.where(logits >= 0,
                -logits + F.softplus(logits, -1, 50),
                -F.softplus(logits, 1, 50))
        probs_gamma = probs ** gamma
        probs_1_gamma = (1. - probs) ** gamma

        ctx.coeff = coeff
        ctx.probs = probs
        ctx.log_probs = log_probs
        ctx.log_1_probs = log_1_probs
        ctx.probs_gamma = probs_gamma
        ctx.probs_1_gamma = probs_1_gamma
        ctx.label = label
        ctx.gamma = gamma

        term1 = probs_1_gamma * log_probs
        term2 = probs_gamma * log_1_probs
        loss = torch.where(label == 1, term1, term2).mul_(coeff).neg_()
        return loss

    @staticmethod
    def backward(ctx, grad_output):
        '''
        compute gradient of focal loss
        '''
        coeff = ctx.coeff
        probs = ctx.probs
        log_probs = ctx.log_probs
        log_1_probs = ctx.log_1_probs
        probs_gamma = ctx.probs_gamma
        probs_1_gamma = ctx.probs_1_gamma
        label = ctx.label
        gamma = ctx.gamma

        term1 = (1. - probs - gamma * probs * log_probs).mul_(probs_1_gamma).neg_()
        term2 = (probs - gamma * (1. - probs) * log_1_probs).mul_(probs_gamma)

        grads = torch.where(label == 1, term1, term2).mul_(coeff).mul_(grad_output)
        return grads, None, None, None


class FocalLossV2(nn.Module):
    '''
    This use better formula to compute the gradient, which has better numeric stability
    '''
    def __init__(self,
                 alpha=0.25,
                 gamma=2,
                 reduction='mean'):
        super(FocalLossV2, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, label):
        loss = FocalSigmoidLossFuncV2.apply(logits, label, self.alpha, self.gamma)
        if self.reduction == 'mean':
            loss = loss.mean()
        if self.reduction == 'sum':
            loss = loss.sum()
        return loss

# version 3: implement wit cpp/cuda to save memory and accelerate
# import focal_cpp # import torch before import cpp extension
# class FocalSigmoidLossFuncV3(torch.autograd.Function):
#     '''
#     use cpp/cuda to accelerate and shrink memory usage
#     '''
#     @staticmethod
#     def forward(ctx, logits, labels, alpha, gamma):
#         logits = logits.float()
#         loss = focal_cpp.focalloss_forward(logits, labels, gamma, alpha)
#         ctx.variables = logits, labels, alpha, gamma
#         return loss
#
#     @staticmethod
#     def backward(ctx, grad_output):
#         '''
#         compute gradient of focal loss
#         '''
#         logits, labels, alpha, gamma = ctx.variables
#         grads = focal_cpp.focalloss_backward(grad_output, logits, labels, gamma, alpha)
#         return grads, None, None, None
#
#
# class FocalLossV3(nn.Module):
#     '''
#     This use better formula to compute the gradient, which has better numeric stability
#     '''
#     def __init__(self,
#                  alpha=0.25,
#                  gamma=2,
#                  reduction='mean'):
#         super(FocalLossV3, self).__init__()
#         self.alpha = alpha
#         self.gamma = gamma
#         self.reduction = reduction
#
#     def forward(self, logits, label):
#         loss = FocalSigmoidLossFuncV3.apply(logits, label, self.alpha, self.gamma)
#         if self.reduction == 'mean':
#             loss = loss.mean()
#         if self.reduction == 'sum':
#             loss = loss.sum()
#         return loss
