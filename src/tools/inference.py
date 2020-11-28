'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 使用膨胀预测的输出实验结果
LastEditTime: 2020-11-28 08:39:03
'''
import sys
sys.path.append("..")

from data.dataloader import make_dataloader
from configs import merage_from_arg,load_arg
from model import build_model
from argparse import ArgumentParser
import torch
import torch.nn as nn
from utils import get_free_device_ids
import copy
import datetime
import os
from PIL import Image
from tqdm import tqdm
import pandas as pd
import numpy as np
import pickle
from os import mkdir,path
Image.MAX_IMAGE_PIXELS = 1000000000000000000
torch.backends.cudnn.benchmark = True



def create_zeros_png(image_w,image_h):
    '''Description:
        0. 先创造一个空白图像，将滑窗预测结果逐步填充至空白图像中；
        1. 填充右下边界，使原图大小可以杯滑动窗口整除；
        2. 膨胀预测：预测时，对每个(1024,1024)窗口，每次只保留中心(512,512)区域预测结果，每次滑窗步长为512，使预测结果不交叠；
    '''
    new_h,new_w = (image_h//1024+1)*1024,(image_w//1024+1)*1024 #填充右边界
    zeros = (256+new_h+256,256+new_w+256)  #填充空白边界
    zeros = np.zeros(zeros,np.uint8)
    return zeros




def tta_forward(dataloader,model,png_shape,device=None):
    image_w,image_h = png_shape
    predict_png = create_zeros_png(image_w,image_h)
    model = model.eval()
    with torch.no_grad():
        for (image,pos_list) in tqdm(dataloader):
            # forward --> predict
            image = image.cuda(device) 

            predict_1 = model(image)

            predict_2 = model(torch.flip(image,[-1]))
            predict_2 = torch.flip(predict_2,[-1])

            predict_3 = model(torch.flip(image,[-2]))
            predict_3 = torch.flip(predict_3,[-2])

            predict_4 = model(torch.flip(image,[-1,-2]))
            predict_4 = torch.flip(predict_4,[-1,-2])

            predict_list = predict_1 + predict_2 + predict_3 + predict_4   
            predict_list = torch.argmax(predict_list.cpu(),1).byte().numpy() # n x h x w
        
            batch_size = predict_list.shape[0] # batch大小
            for i in range(batch_size):
                predict = predict_list[i]
                pos = pos_list[i,:]
                [topleft_x,topleft_y,buttomright_x,buttomright_y] = pos

                if(buttomright_x-topleft_x)==1024 and (buttomright_y-topleft_y)==1024:
                    # 每次预测只保留图像中心(512,512)区域预测结果
                    predict_png[topleft_y+256:buttomright_y-256,topleft_x+256:buttomright_x-256] = predict[256:768,256:768]
                else:
                    raise ValueError("target_size!=512， Got {},{}".format(buttomright_x-topleft_x,buttomright_y-topleft_y))
    
    h,w = predict_png.shape
    predict_png =  predict_png[256:h-256,256:w-256] # 去除整体外边界
    predict_png = predict_png[:image_h,:image_w]    # 去除补全512整数倍时的右下边界
    return predict_png

def label_resize_vis(label, img=None,alpha=0.5):
    '''
    :param label:原始标签 
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    label = cv.resize(label.copy(),None,fx=0.1,fy=0.1)
    r = np.where(label == 1, 255, 0)
    g = np.where(label == 2, 255, 0)
    b = np.where(label == 3, 255, 0)
    yellow = np.where(label == 4, 255, 0)
    anno_vis = np.dstack((b, g, r)).astype(np.uint8)
    # 黄色分量(红255, 绿255, 蓝0)
    anno_vis[:, :, 0] = anno_vis[:, :, 0] + yellow
    anno_vis[:, :, 1] = anno_vis[:, :, 1] + yellow
    anno_vis[:, :, 2] = anno_vis[:, :, 2] + yellow
    if img is None:
        return anno_vis
    else:
        overlapping = cv.addWeighted(img, alpha, anno_vis, 1-alpha, 0)
        return overlapping






if __name__ == "__main__":
    # 若更新了load_arg函数，需要对应更新merage_from_arg()
    arg = vars(load_arg())
    if arg['load_path'] != None: #优先级：arg传入命令 >model中存的cfg > config_file
        state_dict = torch.load(arg['load_path'],map_location='cpu')
        if 'cfg' in state_dict.keys():
            cfg = state_dict['cfg']
    # 待修改
    config_file = arg["CONFIG_FILE"]
    config_file = config_file.replace("../","").replace(".py","").replace('/','.')
    exec(r"from {} import config as cfg".format(config_file))

    # load `model` & `dataloader`
    cfg = merage_from_arg(cfg,arg)

    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%Y%m%d_')
    save_dir = os.path.join(cfg['save_dir'],time_str+cfg['tag'])
    cfg['save_dir'] = save_dir
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("Save_dir :",save_dir)

    model = build_model(cfg,pretrain_path=arg['load_path'])
    # get free_device
    free_device_ids = get_free_device_ids()
    max_num_devices = cfg['max_num_devices']
    if len(free_device_ids)>=max_num_devices:
        free_device_ids = free_device_ids[:max_num_devices]
    # print(free_device_ids)

    master_device = free_device_ids[0]
    model = nn.DataParallel(model,device_ids=free_device_ids).cuda(master_device)

    # save_dir
    curr_time = datetime.datetime.now()
    time_str = datetime.datetime.strftime(curr_time,'%Y%m%d_')
    model_tag = (os.path.split(arg['load_path'])[1]).split('.')[0]
    save_dir = os.path.join(r'../exp/',time_str+model_tag)
    if not os.path.isdir(save_dir):
        os.makedirs(save_dir)
    print("Save_dir :",save_dir)


    image_3_pipeline = make_dataloader(cfg['test_image3_pipeline'])
    image_3_predict = tta_forward(image_3_pipeline,model,device=master_device,png_shape=cfg['test_image3_pipeline']['image_shape'])
    pil_image = Image.fromarray(image_3_predict)
    pil_image.save(os.path.join(save_dir,"image_"+str(3)+"_predict.png"))
    del image_3_pipeline,image_3_predict
    
    image_4_pipeline = make_dataloader(cfg['test_image4_pipeline'])
    image_4_predict = tta_forward(image_4_pipeline,model,device=master_device,png_shape=cfg['test_image4_pipeline']['image_shape'])
    pil_image = Image.fromarray(image_4_predict)
    pil_image.save(os.path.join(save_dir,"image_"+str(4)+"_predict.png"))



    

    