
config = dict(
    # Basic Config
    enable_backends_cudnn_benchmark = True,
    max_epochs = 40+1,
    save_period = 5,
    log_period = 0.05,  # 每隔 int(log_period*len(dataloader)) 打印一次 loss
    save_dir = r"../checkpoints/", # {'model':model.state_dict(),'config':config} 缓存路径
    log_dir = r"../log/",

    # Dataset
    # train_dataloader：
    #           --> dataloader: image2batch,继承自 torch.utils.data.DataLoader,
    #               --> batch_size: 每个batch数目
    #               --> shuffle : 是否打乱数据，默认训练集打乱，测试集不打乱
    #               --> num_workers : 多线程加载数据
    #               --> drop_last : 若 len_epoch 无法整除 batch_size 时，丢弃最后一个batch。（在较早的torch版本中，开启多卡GPU加速后，
    #                               若batch无法整除多卡数目，代码运行会报错，避免出错风险丢弃最后一个batch）
    #
    #           --> transforms: 在线数据增强加载，传入数据增强函数及对应参数配置，相关代码在 data/transforms/opencv_transforms.py
    #
    #           --> dataset: 加载image和label，继承自 torch.utils.data.Dataset,对应代码在 data/dataset/bulid.py
    #                           
    train_pipeline = dict(
        dataloader = dict(batch_size = 8,num_workers = 12,drop_last = True,pin_memory=True,shuffle=True),

        dataset = dict(type="PNG_Dataset",
                    csv_file=r'/home/LinHonghui/Datasets/tianchi_CountyAgricultural/Crop1024/train.csv',
                    image_dir=r'/home/LinHonghui/Datasets/tianchi_CountyAgricultural/Crop1024/image/',
                    mask_dir=r'/home/LinHonghui/Datasets/tianchi_CountyAgricultural/Crop1024/label/'),

        transforms = [
            dict(type="RandomCrop",p=1,output_size=(256,256)),
            dict(type="RandomHorizontalFlip",p=0.5),
            dict(type="RandomVerticalFlip",p=0.5),
            dict(type="ColorJitter",brightness=0.08,contrast=0.08,saturation=0.08,hue=0.08),
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True),
            ],
    ),

    test_image3_pipeline = dict(
        dataloader = dict(batch_size = 42,num_workers = 12,drop_last = False,pin_memory=True,shuffle=False),
        dataset = dict(type="Inference_Dataset",
                    csv_file=r"/home/LinHonghui/Datasets/tianchi_CountyAgricultural/Crop1024/image_3.csv",
                    image_dir=r'/home/LinHonghui/Datasets/tianchi_CountyAgricultural/Crop1024/image/',),
        transforms = [
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True),
            ],
        image_shape = (20767,42614), 
    ),

    test_image4_pipeline = dict(
        dataloader = dict(batch_size = 42,num_workers = 12,drop_last = False,pin_memory=True,shuffle=False),
        dataset = dict(type="Inference_Dataset",
                    csv_file=r"/home/LinHonghui/Datasets/tianchi_CountyAgricultural/Crop1024/image_4.csv",
                    image_dir=r'/home/LinHonghui/Datasets/tianchi_CountyAgricultural/Crop1024/image/',),
        transforms = [
            dict(type="ToTensor",),
            dict(type="Normalize",mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225],inplace=True),
            ],
        image_shape = (29003,35055), 
    ),




    # Model
    # model : 
    ##      --> backbone : 特征提取，需在model/backbone中定义
    ##      --> head : classification heads,model/heads
    ##      --> losses: criterion. model/losses 
    model = dict(
        net = dict(type="deeplabv3plus",num_classes=5),
        backbone = dict(type="resnet101",pretrained=True,replace_stride_with_dilation=[False,False,2]),
        head = dict(type="ASPP",in_channels=2048,out_channels=256,dilation_list=[6,12,18]),
        loss = dict(type="LabelSmoothing",win_size=11,num_classes=5,),
        # loss = dict(type="cross_entropy2d",weight=[10/9,10/9,10/9,10/9,5/9]),

    ),

    metric = dict(type="Label_Accuracy_drop_edge"),

    multi_gpu = True,
    max_num_devices = 1, #自动获取空闲显卡，默认第一个为主卡


    # Solver
    ## lr_scheduler : 学习率调整策略，默认从 torch.optim.lr_scheduler 中加载
    ## optimizer : 优化器，默认从 torch.optim 中加载
    lr_scheduler = dict(type="StepLR",step_size=5,gamma=1/3), # cycle_momentum=False if optimizer==Adam
    optimizer = dict(type="Adam",lr=1e-4,weight_decay=1e-5),

)