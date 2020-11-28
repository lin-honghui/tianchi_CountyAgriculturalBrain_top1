###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2020-11-28 08:34:42
 # @Description : 
### 

cd src/

# config_file="../configs/deeplabv3plus_StepLR_Adam.py"

load_path="../checkpoints/20201128_deeplabv3plus_resnet50_StepLR_Adam/deeplabv3plus_resnet50_StepLR_Adam_temp.pth"
config_file="../configs/deeplabv3plus_resnet50_StepLR_Adam.py"
python tools/inference.py -config_file $config_file -load_path $load_path

