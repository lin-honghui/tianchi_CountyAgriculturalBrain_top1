'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-28 05:43:40
'''
import sys
sys.path.append('..')
from data.dataloader import make_dataloader
from configs import merage_from_arg,load_arg
from model import build_model
from solver import make_optimizer,wrapper_lr_scheduler
from argparse import ArgumentParser
from engine import do_train
import torch.nn as nn
import torch
from utils import get_free_device_ids
import copy
import datetime
import os
# os.environ['CUDA_VISIBLE_DEVICES'] = '2,3'



if __name__ == "__main__":
    # 若更新了load_arg函数，需要对应更新merage_from_arg()
    arg = vars(load_arg())
    # 待修改
    config_file = arg["CONFIG_FILE"]
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')

    exec(r"from {} import config as cfg".format(config_file))
    # if arg['MODEL.LOAD_PATH'] != None: #优先级：arg传入命令 >model中存的cfg > config_file
    #     cfg = torch.load(arg['MODEL.LOAD_PATH'])['cfg']
    cfg = merage_from_arg(cfg,arg)
    print(cfg)
    cfg_copy = copy.deepcopy(cfg)

    train_dataloader = make_dataloader(cfg['train_pipeline'])
    
    
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%Y%m%d_')
    save_dir = os.path.join(cfg['save_dir'],time_str+cfg['tag'])
    log_dir = os.path.join(cfg['log_dir'],"log_"+time_str+cfg['tag'])
    cfg['save_dir'] = save_dir
    cfg['log_dir'] = log_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    if not os.path.isdir(log_dir):
        os.makedirs(log_dir)
    print("Save_dir :",save_dir)
    print("Log_dir :", log_dir)

    # import pdb; pdb.set_trace()
    model = build_model(cfg,pretrain_path=arg['load_path'])

    optimizer = make_optimizer(cfg['optimizer'],model)
    lr_scheduler = wrapper_lr_scheduler(cfg['lr_scheduler'],optimizer)

    if arg['device']: # 传入命令指定 device id
        free_device_ids = arg['device']
    else:
        free_device_ids = get_free_device_ids()

    max_num_devices = cfg['max_num_devices']
    if len(free_device_ids)>=max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]

    master_device = free_device_ids[0]
    model.cuda(master_device)
    model = nn.DataParallel(model,device_ids=free_device_ids).cuda(master_device)


    if cfg['enable_backends_cudnn_benchmark']:
        print("enable backends cudnn benchmark")
        torch.backends.cudnn.benchmark = True



    cfg_copy['save_dir'] = save_dir # 更新存储目录
    cfg_copy['log_dir'] = log_dir # 更新存储目录
    # import pdb; pdb.set_trace()
    do_train(cfg_copy,model=model,train_loader=train_dataloader,val_loader=None,optimizer=optimizer,
                scheduler=lr_scheduler,metrics=None,device=free_device_ids)