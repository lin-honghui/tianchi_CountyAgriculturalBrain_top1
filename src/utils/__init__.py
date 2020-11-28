'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-28 05:48:19
'''
import utils.metric as Metric
from .nvidia_info import *

def make_metrics(cfg):
    if hasattr(Metric,cfg.UTILS.METRICS):
        metrics = getattr(Metric,cfg.UTILS.METRICS)()
        return metrics
    else:
        raise Exception("Invalid metric",cfg.UTILS.METRICS)