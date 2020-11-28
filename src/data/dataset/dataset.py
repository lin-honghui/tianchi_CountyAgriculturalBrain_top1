'''
Author      : now more
Connect     : lin.honghui@qq.com
LastEditors: Please set LastEditors
Description : 
LastEditTime: 2020-11-28 08:37:52
'''

from torch.utils.data import Dataset
from PIL import Image
import pandas as pd
import numpy as np
import cv2 as cv
import os
class PNG_Dataset(Dataset):
    """ Tianchi AI 2019 """
    def __init__(self,csv_file,image_dir,mask_dir,transforms=None):
        '''
        Description: 
        Args (type): 
            csv_file  (string): Path to the file with annotations, see `utils/data_prepare` for more information.
            image_dir (string): Derectory with all images.
            mask_dir (string): Derectory with all labels.
            transforms (callable,optional): Optional transforms to be applied on a sample.
        return: 
        '''
        self.csv_file = pd.read_csv(csv_file,header=None)
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transforms = transforms
    
    def __len__(self):
        return len(self.csv_file)
    
    def __getitem__(self,idx):
        """
        Args:
            idx (int): index of sample
        """
        filename = self.csv_file.iloc[idx,0]
        _,filename = os.path.split(filename)
        image_path = os.path.join(self.image_dir,filename)
        mask_path = os.path.join(self.mask_dir,filename)
        image = Image.open(image_path)
        image = np.asarray(image) #mode:RGBA
        image = cv.cvtColor(image,cv.COLOR_RGBA2RGB) # PIL(RGBA)-->cv2(RGB)

        mask = np.asarray(Image.open(mask_path)) #mode:P(单通道)
        mask = mask.copy()

        sample = {'image':image,'mask':mask}

        if self.transforms:
            sample = self.transforms(sample)

        image,mask = sample['image'],sample['mask']
        return image,mask


class Inference_Dataset(Dataset):
    def __init__(self,image_dir,csv_file,transforms=None):
        '''
        Description: 
        Args (type): 
            csv_file  (string): Path to the file with annotations, see `utils/data_prepare` for more information.
            image_dir (string): Derectory with all images.
            transforms (callable,optional): Optional transforms to be applied on a sample.
        return: 
        '''
        self.image_dir = image_dir
        self.csv_file = pd.read_csv(csv_file,header=None)
        self.transforms = transforms
    def __len__(self):
        return len(self.csv_file)
    def __getitem__(self,idx):
        filename = self.csv_file.iloc[idx,0]
        _,filename = os.path.split(filename)
        image_path = os.path.join(self.image_dir,filename)
        image = np.asarray(Image.open(image_path)) #mode:RGBA
        image = cv.cvtColor(image,cv.COLOR_RGBA2RGB) # PIL(RGBA)-->cv2(RGB)
        
        sample = {'image':image}
        if self.transforms:
            sample = self.transforms(sample)
        image = sample['image']

        
        pos_list = self.csv_file.iloc[idx,1:].values.astype("int")  # ---> (topleft_x,topleft_y,buttomright_x,buttomright_y)
        return image,pos_list

    
