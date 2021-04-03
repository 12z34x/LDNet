import os
from pprint import pprint
from utils import select_device
user_dir = os.path.expanduser('~')
import torch

class Config:
    # data
    dataset = "dronecc"
    img_type = 'jpg'
    root_dir = user_dir + "/data/DroneCC"
    root_dir_ucf=user_dir + "/data/UCF_CC_50"
    root_dir_qnrf=user_dir + "/data/UCF_QNRF"
    root_dir_st_B=user_dir + "/data/ST_CC/part_B"
    root_dir_st_A=user_dir + "/data/ST_CC/part_A"
    root=user_dir+"/data"

    num_classes = 1
    input_size = (1024,512)#if down sample 1024*768(ST)
    norm_cfg = dict(mean=[0.437, 0.446, 0.434], std=[0.206, 0.198, 0.203], para=500)#设置密度权重, 500 drone 800 A
    resume =False #True
    pre ="/home/twsf/work/EvaNet/log/dronecc/20200902_11_train/model_best.pth.tar"

    # model
    backbone = "ghostnet"#'mobilenetv2'#res2net101
    output_stride = 8
    sync_bn = False

    # train
    temporal = True  # 暂时不要设置False
    batch_size = 2 # 8
    epochs = 10000 #64 is enough
    freeze_bn = False #False

    loss_region = dict(
        type="CrossEntropyLoss",
        ignore_index=-1,
        weight=[1,30] )    

    loss_region_st=dict(#平衡正负样本
        type="FocalLoss",
        #reduce=False,
        #class_num=2,#
        alpha = torch.Tensor([0.25]),
        gamma = 2,
        loss_weight=30#设置区域权重
    )
    loss_density = dict(
        type="MSELoss",
        reduction="mean"
    )

    # optimizer
    use_balanced_weights = False#use the region weight
    lr = 0.000001#0.0008
    decay = 5e-4 #0 for adam, 5e-4 for SGD
    betas=(0.9,0.999)#for adam
    #eps=1e-08#for adam
    momentum = 0.9#for SGD
    steps = [0.8, 0.9]
    gamma = 0.95
    workers = 1

    # visual
    print_freq = 150
    plot_every = 9999999999999999  # every n batch plot

    seed = 1

    def _parse(self, kwargs):
        state_dict = self._state_dict()
        for k, v in kwargs.items():
            if k not in state_dict:
                raise ValueError('UnKnown Option: "--%s"' % k)
            setattr(self, k, v)

        self.device, self.gpu_id = select_device()

        print('======user config========')
        pprint(self._state_dict())
        print('==========end============')

    def _state_dict(self):
        return {k: getattr(self, k) for k, _ in Config.__dict__.items()
                if not k.startswith('_')}

opt = Config()
