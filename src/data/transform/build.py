'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 根据cfg配置文件，加载数据增强函数
LastEditTime: 2020-11-27 04:38:35
'''
from . import opencv_transforms as transforms

def build_transforms(cfg_transforms):
    cfg_transforms = cfg_transforms.copy()
    transforms_list = list()
    for item in cfg_transforms:
        transforms_type = item.pop("type")
        transforms_kwags = item
        if hasattr(transforms,transforms_type):
            transforms_list.append(getattr(transforms,transforms_type)(**transforms_kwags))
        else:
            raise ValueError("\'type\' of transforms is not defined. Got {}".format(transforms_type))
    print(transforms_list)
    return transforms.Compose(transforms_list)