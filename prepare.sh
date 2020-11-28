###
 # @Author      : now more
 # @Contact     : lin.honghui@qq.com
 # @LastEditors: Please set LastEditors
 # @LastEditTime: 2020-11-28 09:03:56
 # @Description : 
### 

## 0.

## 1. data prepare
save_dir="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/Crop1024"

image_10_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_train_20190726/image_10.png"
label_10_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_train_20190726/image_10_label.png"
image_11_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_train_20190726/image_11.png"
label_11_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_train_20190726/image_11_label.png"
image_20_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_train_20190726/image_20.png"
label_20_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_train_20190726/image_20_label.png"
image_21_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_train_20190726/image_21.png"
label_21_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_train_20190726/image_21_label.png"

image_3_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_test_a_20190726/image_3.png"
image_4_path="/home/LinHonghui/Datasets/tianchi_CountyAgricultural/jingwei_round2_test_a_20190726/image_4.png"

python src/utils/data_prepare.py -image_path $image_10_path -label_path $label_10_path -save_dir $save_dir
python src/utils/data_prepare.py -image_path $image_11_path -label_path $label_11_path -save_dir $save_dir
python src/utils/data_prepare.py -image_path $image_20_path -label_path $label_20_path -save_dir $save_dir
python src/utils/data_prepare.py -image_path $image_21_path -label_path $label_21_path -save_dir $save_dir

python src/utils/concat_csv.py -root_dir $save_dir

python src/utils/data_prepare.py -image_path $image_3_path -save_dir $save_dir
python src/utils/data_prepare.py -image_path $image_4_path -save_dir $save_dir





