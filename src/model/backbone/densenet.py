'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-11-28 07:50:11
Description : 
'''
import torch
import torch.nn as nn
import torchvision.models.densenet as densenet
import torch.nn.functional as F






# class densenet121(nn.Module):  
#     def __init__(self,pretrained=True, progress=True, **kwargs):
#         super(densenet121,self).__init__()
#         backbone = densenet.densenet121(pretrained,progress,**kwargs)
#         self.backbone = backbone.features
#     def forward(self,x):
#         out = F.relu(self.backbone(x), inplace=True)
#         return out # 1024 channel

# class densenet169(nn.Module):
#     def __init__(self,pretrained=True, progress=True, **kwargs):
#         super(densenet169,self).__init__()
#         backbone = densenet.densenet169(pretrained,progress,**kwargs)
#         self.backbone = backbone.features
#     def forward(self,x):
#         out = F.relu(self.backbone(x), inplace=True)
#         return out # 1664 channel






if __name__ == "__main__":
    pass
