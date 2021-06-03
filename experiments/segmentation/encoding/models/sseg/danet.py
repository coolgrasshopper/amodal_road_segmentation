
from __future__ import division
import os
import numpy as np
import torch
import torch.nn as nn
from torch.nn import functional as F
from torch.nn.functional import upsample,normalize
from ...nn import PAM_Module
from ...nn import CAM_Module
from .base import BaseNet
from torch.nn.parallel.scatter_gather import gather


__all__ = ['DANet', 'get_danet']
def random_rectangle_mask_gene(img_size, batch_size, num_rect, device, max_size_factor):

    msk = torch.ones((batch_size, 1, img_size[0], img_size[1]), device=device).float()

    for batch_idx in range(batch_size):
        for i in range(num_rect):
            pos = np.random.rand(2)
            size = np.maximum(np.random.rand(2), 0.1/max_size_factor) * max_size_factor

            h_start = int(pos[0] * img_size[0])
            w_start = int(pos[1] * img_size[1])
            h = np.minimum(int(size[0] * img_size[0]), img_size[0]-h_start)
            w = np.minimum(int(size[1] * img_size[1]), img_size[1]-w_start)

            msk[batch_idx, :, h_start:h_start+h, w_start:w_start+w] = 0.

    return msk


#maxpooling as inpainting module
def MaxpoolingAsInpainting(x, x_fore_augment):
    #print(x.detach().cpu().numpy().shape)
    #print(str(4))
    x_fore_augments = gather(x_fore_augment, 0, dim=0)
    pred = torch.max(x_fore_augment[0], 1)[1].unsqueeze(1)

    #print(str(5))
    #print(x_fore_augment[0].detach().cpu().numpy().shape)

    x_fore_pool = F.interpolate(pred.float(), size=x.size()[2:4]).detach().clone()
    fore_msk = (x_fore_pool > 0.5).float().detach().clone()
    fore_msk = F.max_pool2d((fore_msk).detach().clone(), kernel_size=3, stride=1, padding=1)
    bkgd_msk_prev = (1.-fore_msk).detach().clone()
    bkgd_msk =  bkgd_msk_prev.detach().clone()
    x_new = x.detach().clone() * bkgd_msk.detach()[0].clone()
    x_patch = x.detach().clone() * 0.

    while torch.mean(1.-bkgd_msk).item() > 0.0001 and torch.min(torch.mean(bkgd_msk, dim=(1, 2, 3))) > 0.:
        x_new =0.5* F.max_pool2d((x_new).detach().clone(), kernel_size=3, stride=1, padding=1)+ 0.5*F.avg_pool2d((x_new).detach().clone(), kernel_size=3, stride=1, padding=1)
        bkgd_msk = F.max_pool2d((bkgd_msk_prev).detach().clone(),kernel_size=3, stride=1, padding=1)
        x_patch = (x_patch + (bkgd_msk-bkgd_msk_prev)*x_new).detach().clone()
        bkgd_msk_prev = bkgd_msk.detach().clone()

    return x_patch,pred



class PSPModule(nn.Module):
    def __init__(self, features, out_features=1024, sizes=(1, 2, 3, 6)):
        super().__init__()
        self.stages = []
        self.stages = nn.ModuleList([self._make_stage(features, size) for size in sizes])
        self.bottleneck = nn.Conv2d(features * (len(sizes) + 1), out_features, kernel_size=1)
        self.relu = nn.ReLU()

    def _make_stage(self, features, size):
        prior = nn.AdaptiveAvgPool2d(output_size=(size, size))
        conv = nn.Conv2d(features, features, kernel_size=1, bias=False)
        return nn.Sequential(prior, conv)

    def forward(self, feats):
        h, w = feats.size(2), feats.size(3)
        priors = [F.interpolate(input=stage(feats), size=(h, w), mode='bilinear', align_corners=False) for stage in self.stages] + [feats]
        bottle = self.bottleneck(torch.cat(priors, 1))
        return self.relu(bottle)

class PSPUpsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=False)
        return self.conv(p)


class PSPNet(nn.Module):
    def __init__(self, n_classes, sizes, psp_size, backend,pretrained):
        super().__init__()
        self.feats = backend
        self.psp = PSPModule(psp_size, 1024, sizes)
        self.drop_1 = nn.Dropout2d(p=0.3)

        self.up_1 = PSPUpsample(1024, 256)
        self.up_2 = PSPUpsample(256, 64)
        self.up_3 = PSPUpsample(64, 64)

        self.drop_2 = nn.Dropout2d(p=0.15)
        self.final = nn.Sequential(
            nn.Conv2d(64, n_classes, kernel_size=1),
        )

    def forward(self, x, training, device, gt_fore, use_gt_fore):

        f = self.feats
        p = self.psp(f)
        p = self.drop_1(p)

        p = self.up_1(p)
        p = self.drop_2(p)

        p = self.up_2(p)
        p = self.drop_2(p)

        p = self.up_3(p)
        p = self.drop_2(p)

        return self.final(p)

class Upsample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.PReLU()
        )

    def forward(self, x):
        h, w = 2 * x.size(2), 2 * x.size(3)
        p = F.interpolate(input=x, size=(h, w), mode='bilinear', align_corners=False)
        return self.conv(p)


def load_weights_sequential(target, source_state):
    model_to_load= {k: v for k, v in source_state.items() if k in target.state_dict().keys()}
    target.load_state_dict(model_to_load, strict=False)


def conv3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, dilation=dilation, bias=False)


class DANet(BaseNet):
    r"""Fully Convolutional Networks for Semantic Segmentation

    Parameters
    ----------
    nclass : int
        Number of categories for the training dataset.
    backbone : string
        Pre-trained dilated backbone network type (default:'resnet50'; 'resnet50',
        'resnet101' or 'resnet152').
    norm_layer : object
        Normalization layer used in backbone network (default: :class:`mxnet.gluon.nn.BatchNorm`;


    Reference:

        Long, Jonathan, Evan Shelhamer, and Trevor Darrell. "Fully convolutional networks
        for semantic segmentation." *CVPR*, 2015

    """
    def __init__(self, nclass, backbone, aux=False, se_loss=False, norm_layer=nn.BatchNorm2d, **kwargs):
        super(DANet, self).__init__(nclass, backbone, aux, se_loss, norm_layer=norm_layer, **kwargs)
        #print(str(10))
        inter_channels=512
        self.head = DANetHead(2048, nclass, norm_layer)
        self.conv511 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())


        #self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        #self.conv10 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(512, 2, 1))

        self.convforeout = nn.Sequential(
            #nn.Conv2d(1024, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Upsample(512, 256),
            Upsample(256, 128),
            nn.Conv2d(128, 1, 1, bias=False)
        )

        self.convforeout2 = nn.Sequential(
            #nn.Conv2d(64, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Upsample(512, 256),
            Upsample(256, 128),
            nn.Conv2d(128, 1, 1, bias=False)
        )
        self.convforeout3 = nn.Sequential(
            #nn.Conv2d(64, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, 3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            Upsample(512, 256),
            Upsample(256, 128),
            nn.Conv2d(128, 1, 1, bias=False)
        )


    def forward(self, x):
        #print(str(0))
        imsize = x.size()[2:]
        _, _, c3, c4 = self.base_forward(x)
        #print(str(100))
        x = self.head(c4)
        x = list(x)

        #x[0] = upsample(x[0], imsize, **self._up_kwargs)
        x[0]=self.conv511(x[0])
        x[1] = upsample(x[1], imsize, **self._up_kwargs)
        x[2] = upsample(x[2], imsize, **self._up_kwargs)
        x[3] = upsample(x[3], imsize, **self._up_kwargs)
        #2.maxpooling as inpainting
        final_featuremap,x_fore_augment=MaxpoolingAsInpainting(x[0],[x[1],x[2],x[3]])




        x1 = final_featuremap * (F.interpolate(x_fore_augment.float(), size=x[0].size()[2:4]) > 0.5).float() +\
            x[0] * (F.interpolate(x_fore_augment.float(), size=x[0].size()[2:4]) < 0.5).float()

        #=output=self.conv10(x1)

        x_fore_pred = self.convforeout(x1+x[0])
        x_fore_pred = torch.sigmoid(x_fore_pred)
        output = F.interpolate(x_fore_pred, scale_factor=2)

        #output = upsample(output, imsize, **self._up_kwargs)

        outputs = [output]
        outputs.append(x[1])
        outputs.append(x[2])
        outputs.append(x[3])
        outputs.append(F.interpolate(torch.sigmoid(self.convforeout2(x[0])),scale_factor=2))
        outputs.append(F.interpolate(torch.sigmoid(self.convforeout3(x1)),scale_factor=2))
        #print("output length is:"+str(len(outputs))+"\n")
        return tuple(outputs)

class DANetHead(nn.Module):
    def __init__(self, in_channels, out_channels, norm_layer):
        super(DANetHead, self).__init__()
        inter_channels = in_channels // 4
        self.conv5a = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv5c = nn.Sequential(nn.Conv2d(in_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.sa = PAM_Module(inter_channels)

        self.sc = CAM_Module(inter_channels)
        self.conv51 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())
        self.conv52 = nn.Sequential(nn.Conv2d(inter_channels, inter_channels, 3, padding=1, bias=False),
                                   norm_layer(inter_channels),
                                   nn.ReLU())

        self.conv6 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv7 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))

        self.conv8 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, out_channels, 1))
        self.conv9 = nn.Sequential(nn.Dropout2d(0.1, False), nn.Conv2d(inter_channels, 2, 1))

        #self.layer3 = self._make_layer(block, 256, layers[2], stride=1, dilation=2)
        #self.layer4 = self._make_layer(block, 512, layers[3], stride=1, dilation=4)


    def _make_layer(self, block, planes, blocks, stride=1, dilation=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))

        return nn.Sequential(*layers)

    def forward(self, x):
        #print(str(2))
        feat1 = self.conv5a(x)



        #the foreground branch
        sa_feat = self.sa(feat1)
        sa_conv = self.conv51(sa_feat)
        sa_output = self.conv6(sa_conv)

        feat2 = self.conv5c(x)
        sc_feat = self.sc(feat2)
        sc_conv = self.conv52(sc_feat)
        sc_output = self.conv7(sc_conv)

        feat_sum = sa_conv+sc_conv

        sasc_output = self.conv8(feat_sum)
        #print(str(3))
        #the road branch
        #1. intermediate feature map extraction
        sa_feat2=self.sa(feat1)
        #print(str(1))

        #x_3 = self.layer3(x)
        #x = self.layer4(x_3)
        #final=PSPNet(x,, psp_size=1024, pretrained=False, n_classes=3)
        #use pspnet maybe?
        return (sa_feat2, sasc_output, sa_output, sc_output)

def get_danet(dataset='pascal_voc', backbone='resnet50', pretrained=False,
           root='~/.encoding/models', **kwargs):
    r"""DANet model from the paper `"Dual Attention Network for Scene Segmentation"
    <https://arxiv.org/abs/1809.02983.pdf>`
    """
    acronyms = {
        'pascal_voc': 'voc',
        'pascal_aug': 'voc',
        'pcontext': 'pcontext',
        'ade20k': 'ade',
        'cityscapes': 'cityscapes',
    }
    # infer number of classes
    from ...datasets import datasets, VOCSegmentation, VOCAugSegmentation, ADE20KSegmentation
    model = DANet(20, backbone=backbone, root=root, **kwargs)
    if pretrained:
        from .model_store import get_model_file
        model.load_state_dict(torch.load(
            get_model_file('fcn_%s_%s'%(backbone, acronyms[dataset]), root=root)),
            strict=False)
    return model
