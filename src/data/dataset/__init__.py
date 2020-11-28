'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-27 04:10:57
'''

import data.dataset.dataset as Datasets
from copy import deepcopy

def build_dataset(cfg_dataset,transforms=None):
    '''
    Description: 
    Args (type): 
        cfg_dataset (dict): 
        transforms (callable,optional): Optional transforms to be applied on a sample.
    return: 
        dataset(torch.utils.data.Dataset)
    '''
    cfg_dataset = deepcopy(cfg_dataset)
    dataset_type = cfg_dataset.pop('type')
    dataset_kwags = cfg_dataset
    
    if hasattr(Datasets,dataset_type):
        dataset = getattr(Datasets,dataset_type)(**dataset_kwags,transforms=transforms)
    else:
        raise ValueError("\'type\' of dataset is not defined. Got {}".format(dataset_type))

    return dataset


