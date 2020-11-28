###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2020-11-28 08:31:27
 # @Description : 
### 

cd src/

# config_file="../configs/deeplabv3plus_StepLR_Adam.py"
config_file="../configs/deeplabv3plus_resnet50_StepLR_Adam.py"
python tools/train.py -config_file $config_file