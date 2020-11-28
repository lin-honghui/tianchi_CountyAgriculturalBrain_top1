'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-28 07:46:09
'''
import src.model.backbone as Backbones
import src.model.head as Heads
import src.model.loss as Losses

from copy import deepcopy
import torch.nn as nn
import torch

def build_backbone(cfg_backbone):
    backbone_type = cfg_backbone.pop('type')
    if hasattr(Backbones,backbone_type):
        backbone = getattr(Backbones,backbone_type)(**cfg_backbone)
        return backbone
    else:
        raise ValueError("\'type\' of backbone is not defined. Got {}".format(backbone_type))

def build_head(cfg_head):
    head_type = cfg_head.pop('type')
    if hasattr(Heads,head_type):
        head = getattr(Heads,head_type)(**cfg_head)
        return head
    else:
        raise ValueError("\'type\' of head is not defined. Got {}".format(head_type))

def build_loss(cfg_loss):
    loss_type = cfg_loss.pop('type')
    if hasattr(Losses,loss_type):
        criterion = getattr(Losses,loss_type)(**cfg_loss)
        return criterion
    else:
        raise ValueError("\'type\' of loss is not defined. Got {}".format(loss_type))
        


class deeplabv3plus(nn.Module):
    def __init__(self,cfg):
        super(deeplabv3plus,self).__init__()
        cfg_model = deepcopy(cfg['model'])
        cfg_backbone = cfg_model['backbone']
        cfg_loss = cfg_model['loss']
        cfg_head = cfg_model['head']

        self.num_classes = cfg_model['net']['num_classes']        
        self.backbone = build_backbone(cfg_backbone)
        self.head = build_head(cfg_head)
        self.loss = build_loss(cfg_loss)

        self.out1 = nn.Sequential(nn.Conv2d(in_channels=256,out_channels=256,kernel_size=1,stride=1),nn.ReLU())
        self.dropout1 = nn.Dropout(0.5)
        self.up4 = nn.Upsample(scale_factor=4)
        self.up2 = nn.Upsample(scale_factor=2)


        self.conv1x1 = nn.Sequential(nn.Conv2d(2048,256,1,bias=False),nn.ReLU())
        self.conv3x3 = nn.Sequential(nn.Conv2d(512,self.num_classes,1),nn.ReLU())
        self.dec_conv = nn.Sequential(nn.Conv2d(256,256,3,padding=1),nn.ReLU())
        
        

        
    def forward(self,x,targets=None):
        x = self.backbone(x)
        out1 = self.head(x)
        out1 = self.out1(out1)
        out1 = self.dropout1(out1)
        out1 = self.up4(out1)

        dec = self.conv1x1(x)
        dec = self.dec_conv(dec)
        dec = self.up4(dec)
        
        contact = torch.cat((out1,dec),dim=1)
        out = self.conv3x3(contact)
        out = self.up4(out)

        if self.training:
            loss = self.loss(out,targets)
            return loss
        else:
            return out
