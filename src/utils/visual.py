'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-11-28 08:58:43
Description : 
'''

import numpy as np
import cv2 as cv
from PIL import Image
import os
Image.MAX_IMAGE_PIXELS = 1000000000000

def label_resize_vis(mask_path,image_path,alpha=0.5):
    '''
    :param label:原始标签 
    :param img: 原始图像
    :param alpha: 透明度
    :return: 可视化标签
    '''
    label = cv.resize(cv.imread(mask_path,cv.IMREAD_GRAYSCALE),None,fx=0.1,fy=0.1)
    image = cv.resize(cv.imread(image_path),None,fx=0.1,fy=0.1)
    r = np.where(label == 1, 255, 0)
    g = np.where(label == 2, 255, 0)
    b = np.where(label == 3, 255, 0)
    yellow = np.where(label == 4, 255, 0)
    anno_vis = np.dstack((b, g, r)).astype(np.uint8)
    # 黄色分量(红255, 绿255, 蓝0)
    anno_vis[:, :, 0] = anno_vis[:, :, 0] + yellow
    anno_vis[:, :, 1] = anno_vis[:, :, 1] + yellow
    anno_vis[:, :, 2] = anno_vis[:, :, 2] + yellow
    if image is None:
        return anno_vis
    else:
        overlapping = cv.addWeighted(image, alpha, anno_vis, 1-alpha, 0)
        return overlapping
        
if __name__ == "__main__":
    predict_dir = r"/home/LinHonghui/Project/tianchi_CountyAgriculturalBrain_top1/exp/20201128_deeplabv3plus_resnet50_StepLR_Adam_temp"
    image_3_path = r"/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_test_a_20190726/image_3.png"
    image_4_path = r"/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_test_a_20190726/image_4.png"

    image_3_predict_path = os.path.join(predict_dir,"image_3_predict.png")
    result = label_resize_vis(image_3_predict_path,image_3_path)
    cv.imwrite(os.path.join(predict_dir,"vis_image_3.png"),result)

    # image_4_predict_path = os.path.join(predict_dir,"image_4_predict.png")
    # result = label_resize_vis(image_4_predict_path,image_4_path)
    # cv.imwrite(os.path.join(predict_dir,"vis_image_4.png"),result)
