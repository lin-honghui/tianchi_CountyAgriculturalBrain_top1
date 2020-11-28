'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-11-28 05:04:41
Description : 
'''
from __future__ import print_function, division
import torch
import torch.nn as nn

class cross_entropy2d(nn.Module):
    def __init__(self,weight=[10/9,10/9,10/9,10/9,5/9]):
        super(cross_entropy2d,self).__init__()
        self.weight = weight
    def forward(self,input, target,size_average=True):
        weight = torch.tensor(self.weight,device=target.device)
        # print(input.shape,target.shape)
        # input: (n, c, h, w), target: (n, h, w)
        n, c, h, w = input.size()
        # log_p: (n, c, h, w)
        log_p = nn.functional.log_softmax(input, dim=1)
        # log_p: (n*h*w, c)
        log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous()
        log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
        log_p = log_p.view(-1, c)
        # target: (n*h*w,)
        mask = target >= 0
        target = target[mask]
        loss = nn.functional.nll_loss(log_p, target.long(), weight=weight, reduction='sum')
        if size_average:
            loss /= mask.data.sum()
        return loss

