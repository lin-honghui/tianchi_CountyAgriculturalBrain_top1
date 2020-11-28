'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-11-28 06:06:31
Description : 
'''
import torch
import torch.nn as nn
import torch.nn.functional as F

class ASPP(nn.Module):
    def __init__(self,in_channels,out_channels=256,dilation_list=[6,12,18]):
        super(ASPP,self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Sequential(nn.Conv2d(in_channels,out_channels,1,padding = 0,dilation=1,bias=False),
                                nn.BatchNorm2d(out_channels))
        self.conv2 = nn.Sequential(nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 3,stride=1,padding = dilation_list[0],dilation = dilation_list[0],bias=False),
                                nn.BatchNorm2d(out_channels))
        self.conv3 = nn.Sequential(nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 3,stride=1,padding = dilation_list[1],dilation = dilation_list[1],bias=False),
                                nn.BatchNorm2d(out_channels))
        self.conv4 = nn.Sequential(nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 3,stride=1,padding = dilation_list[2],dilation = dilation_list[2],bias=False),
                                nn.BatchNorm2d(out_channels))
        self.conv5 = nn.Sequential(nn.Conv2d(in_channels = in_channels,out_channels = out_channels,kernel_size = 1,stride=1,padding = 0,dilation=1,bias=False),
                                nn.BatchNorm2d(out_channels))
        self.adapool = nn.AdaptiveAvgPool2d(1)

        self.convf = nn.Sequential(nn.Conv2d(in_channels = out_channels * 5,out_channels = out_channels,kernel_size = 1,stride=1,padding = 0,dilation=1,bias=False),
                                nn.BatchNorm2d(out_channels))
    def forward(self,x):
        x1 = self.relu(self.conv1(x))
        x2 = self.relu(self.conv2(x))
        x3 = self.relu(self.conv3(x))
        x4 = self.relu(self.conv4(x))
        x5 = self.relu(self.conv5(self.adapool(x)))
        x5 = F.interpolate(x5, size = tuple(x4.shape[-2:]), mode='bilinear')
        x = torch.cat((x1,x2,x3,x4,x5), dim = 1) #channels first
        x = self.relu(self.convf(x))
        return x