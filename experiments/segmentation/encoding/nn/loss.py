import torch
import torch.nn.functional as F
import torch.nn as nn
from torch.autograd import Variable
import kornia
import cv2

__all__ = ['LabelSmoothing', 'NLLMultiLabelSmooth', 'SegmentationLosses']

def dice_loss(yhat, ytrue, epsilon=1e-6):
    """
    Computes a soft Dice Loss

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        epsilon (Float): smoothing value to avoid division by 0
    output:
        DL value with `mean` reduction
    """
    # compute Dice components
    intersection = torch.sum(yhat * ytrue, (1,2,3))
    cardinal = torch.sum(yhat + ytrue, (1,2,3))

    return torch.mean(1. - (2 * intersection / (cardinal + epsilon)))

def tversky_index(yhat, ytrue, alpha=0.3, beta=0.7, epsilon=1e-6):
    """
    Computes Tversky index

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight for False positive
        beta (Float): weight for False negative
                    `` alpha and beta control the magnitude of penalties and should sum to 1``
        epsilon (Float): smoothing value to avoid division by 0
    output:
        tversky index value
    """
    TP = torch.sum(yhat * ytrue, (1,2,3))
    FP = torch.sum((1. - ytrue) * yhat, (1,2,3))
    FN = torch.sum((1. - yhat) * ytrue, (1,2,3))

    return TP/(TP + alpha * FP + beta * FN + epsilon)

def tversky_loss(yhat, ytrue):
    """
    Computes tversky loss given tversky index

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
    output:
        tversky loss value with `mean` reduction
    """
    return torch.mean(1 - tversky_index(yhat, ytrue))

def tversky_focal_loss(yhat, ytrue, alpha=0.7, beta=0.3, gamma=0.75):
    """
    Computes tversky focal loss for highly umbalanced data
    https://arxiv.org/pdf/1810.07842.pdf

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight for False positive
        beta (Float): weight for False negative
                    `` alpha and beta control the magnitude of penalties and should sum to 1``
        gamma (Float): focal parameter
                    ``control the balance between easy background and hard ROI training examples``
    output:
        tversky focal loss value with `mean` reduction
    """

    return torch.mean(torch.pow(1 - tversky_index(yhat, ytrue, alpha, beta), gamma))

def focal_loss(yhat, ytrue, alpha=0.75, gamma=2):
    """
    Computes α-balanced focal loss from FAIR
    https://arxiv.org/pdf/1708.02002v2.pdf

    Args:
        yhat (Tensor): predicted masks
        ytrue (Tensor): targets masks
        alpha (Float): weight to balance Cross entropy value
        gamma (Float): focal parameter
    output:
        loss value with `mean` reduction
    """

    # compute the actual focal loss
    focal = -alpha * torch.pow(1. - yhat, gamma) * torch.log(yhat)
    f_loss = torch.sum(ytrue * focal, dim=1)

    return torch.mean(f_loss)


class LabelSmoothing(nn.Module):
    """
    NLL loss with label smoothing.
    """
    def __init__(self, smoothing=0.1):
        """
        Constructor for the LabelSmoothing module.
        :param smoothing: label smoothing factor
        """
        super(LabelSmoothing, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        logprobs = torch.nn.functional.log_softmax(x, dim=-1)

        nll_loss = -logprobs.gather(dim=-1, index=target.unsqueeze(1))
        nll_loss = nll_loss.squeeze(1)
        smooth_loss = -logprobs.mean(dim=-1)
        loss = self.confidence * nll_loss + self.smoothing * smooth_loss
        return loss.mean()

class NLLMultiLabelSmooth(nn.Module):
    def __init__(self, smoothing = 0.1):
        super(NLLMultiLabelSmooth, self).__init__()
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing

    def forward(self, x, target):
        if self.training:
            x = x.float()
            target = target.float()
            logprobs = torch.nn.functional.log_softmax(x, dim = -1)

            nll_loss = -logprobs * target
            nll_loss = nll_loss.sum(-1)

            smooth_loss = -logprobs.mean(dim=-1)

            loss = self.confidence * nll_loss + self.smoothing * smooth_loss

            return loss.mean()
        else:
            return torch.nn.functional.cross_entropy(x, target)

class SegmentationLosses(nn.CrossEntropyLoss):
    """2D Cross Entropy Loss with Auxilary Loss"""
    def __init__(self, se_loss=False, se_weight=0.2, nclass=-1,
                 aux=False, aux_weight=0.4, weight=None,
                 ignore_index=-1):
        super(SegmentationLosses, self).__init__(weight, None, ignore_index)
        self.se_loss = se_loss
        self.aux = aux
        self.nclass = nclass
        self.se_weight = se_weight
        self.aux_weight = aux_weight
        self.bceloss = nn.BCELoss(weight)

    def forward(self, *inputs):
        if not self.se_loss and not self.aux:
            pred1, pred2, pred3, pred4, pred5,pred6,target1, target2 = inputs
            #background segmentation
            #loss1 = super(SegmentationLosses, self).forward(pred1, target1)

            #foreground segmentation
            loss2 = super(SegmentationLosses, self).forward(pred2, target2)
            loss3 = super(SegmentationLosses, self).forward(pred3, target2)
            loss4 = super(SegmentationLosses, self).forward(pred4, target2)

            loss_entropy_foregd = F.binary_cross_entropy(pred1,target1)
            loss5=F.binary_cross_entropy(pred5,target1)+F.binary_cross_entropy(pred6,target1)

            xl1 = F.interpolate((pred1 > 0.5).float(), scale_factor=1).int().float()*255


            return  loss2+loss3+loss4+loss_entropy_foregd+loss5

        elif not self.se_loss:
            pred1, pred2, target = tuple(inputs)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            return loss1 + self.aux_weight * loss2

        elif not self.aux:
            pred, se_pred, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred)
            loss1 = super(SegmentationLosses, self).forward(pred, target)
            loss2 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.se_weight * loss2

        else:
            pred1, se_pred, pred2, target = tuple(inputs)
            se_target = self._get_batch_label_vector(target, nclass=self.nclass).type_as(pred1)
            loss1 = super(SegmentationLosses, self).forward(pred1, target)
            loss2 = super(SegmentationLosses, self).forward(pred2, target)
            loss3 = self.bceloss(torch.sigmoid(se_pred), se_target)
            return loss1 + self.aux_weight * loss2 + self.se_weight * loss3

    @staticmethod
    def _get_batch_label_vector(target, nclass):
        # target is a 3D Variable BxHxW, output is 2D BxnClass
        batch = target.size(0)
        tvect = Variable(torch.zeros(batch, nclass))
        for i in range(batch):
            hist = torch.histc(target[i].cpu().data.float(),
                               bins=nclass, min=0,
                               max=nclass-1)
            vect = hist>0
            tvect[i] = vect
        return tvect
