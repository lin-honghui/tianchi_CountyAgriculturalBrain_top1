'''
Author      : now more
Contact     : lin.honghui@qq.com
LastEditors: Please set LastEditors
LastEditTime: 2020-11-28 04:04:40
Description : 
'''
import os
import numpy as np
import pandas as pd
from argparse import ArgumentParser


if __name__ == "__main__":
    parser = ArgumentParser(description="")
    parser.add_argument("-root_dir",type=str)
    arg = parser.parse_args()
    root_dir = arg.root_dir
    save_dir = root_dir

    image_10_csv = pd.read_csv(os.path.join(root_dir,'image_10.csv'),header=None)
    image_11_csv = pd.read_csv(os.path.join(root_dir,'image_11.csv'),header=None)
    image_20_csv = pd.read_csv(os.path.join(root_dir,'image_20.csv'),header=None)
    image_21_csv = pd.read_csv(os.path.join(root_dir,'image_21.csv'),header=None)

    total_csv = pd.concat((image_10_csv,image_11_csv,image_20_csv,image_21_csv),axis=0)

    total_csv.to_csv(os.path.join(save_dir,"train.csv"),header=None,index=None)