# tianchi_CountyAgriculturalBrain_top1
天池 2019县域农业大脑AI挑战赛 1/1520 冠军

## 0. 环境
```
conda env create -f env.yaml
conda activate pytorch
```

## 1. 数据预处理
```
sh prepare.sh
```

## 2. 代码目录
```
.
├── checkpoints     (保存训练过程权重)
├── configs         (配置文件)
├── exp             (推理结果保存)
├── jupyter
├── log
├── src
│   ├── data
│   ├── engine
│   ├── model
│   ├── solver
│   ├── tools
│   └── utils
├── prepare.sh
├── inference.sh
├── train.sh
└── README.md

```

## 3. 算法说明
详细方案请见[zhihu](https://zhuanlan.zhihu.com/p/166435221)

线上demo[天池7号馆](https://tianchi.aliyun.com/museum7/?spm=5176.14046517.J_9711814210.24.330d3178iIJT5o#/newprodetail?productId=4)




