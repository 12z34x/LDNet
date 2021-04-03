
import os
import cv2
import shutil
import argparse
import numpy as np
import os.path as osp
from tqdm import tqdm
import matplotlib.pyplot as plt
import h5py
from configs.config import opt
from models import CSMNets as Model
from dataloaders import data_process as dtf
import time
import torch
from torchvision import transforms
# import multiprocessing
# multiprocessing.set_start_method('spawn', True)
user_dir = os.path.expanduser('~')
balance=100
root_dir = user_dir + "/data/DroneCC"
root_dir_ucf=user_dir + "/data/UCF_CC_50"
root_dir_st_B=user_dir + "/data/ST_CC/part_B"
root_dir_st_A=user_dir + "/data/ST_CC/part_A"
root_dir_qnrf=user_dir+"/data/UCF_QNRF"
test_size=(1024,512)
samples=255
noise=0.0
import cv2
fil = np.array([[ 1, 1, 1,1],       #这个是设置的滤波，也就是卷积核
                [ 1, 1, 1,1],
                [ 1, 1, 1,1],
                [ 1, 1, 1,1]])

def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--chekpoint', type=str, default="/home/twsf/work/EvaNet/log/dronecc/20201018_10_train/model_best.pth.tar")
    #"/home/twsf/work/EvaNet/log/dronecc/20200827_19_train/model_best.pth.tar" A
    #"/home/twsf/work/EvaNet/log/dronecc/20200903_21_train/model_best.pth.tar" vis
    #"/home/twsf/work/EvaNet/log/dronecc/20200831_13_train/model_best.pth.tar" qnrf
    #"/home/twsf/work/EvaNet/log/dronecc/20200824_21_train/model_best.pth.tar" ucf

    parser.add_argument('--img_root', type=str, default=root_dir+"/sequences")#change to augment
    parser.add_argument('--img_list', type=str, default=root_dir+"/vallist.txt")
    parser.add_argument('--results_dir', type=str, default="./results")
    parser.add_argument('--show', type=bool, default=False)
    args = parser.parse_args()
    return args

def show_image(img, m):
    args = parse_args()
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1).imshow(img)
    plt.subplot(2, 2, 2).imshow(m, cmap=cm.jet)
    plt.savefig(args.db_root+'mask_image.jpg')
    #plt.show()

args = parse_args()
opt._parse({})


def test():

    mask_size = (int(test_size[0]/8),int(test_size[1]/8))
    if osp.exists(args.results_dir):
        shutil.rmtree(args.results_dir)
    os.makedirs(args.results_dir)

    # data
    imgs_path = []
    with open(args.img_list, 'r') as f:
        for dir_name in f.readlines():
            img_dir = osp.join(args.img_root, dir_name.strip())
            # if not osp.exists(img_dir):
            #     os.mkdir(img_dir)
            for img_name in os.listdir(img_dir):
                imgs_path.append(osp.join(img_dir, img_name))

    transform = transforms.Compose([
        dtf.FixedNoMaskResize(size=test_size),  # resize image
        dtf.Normalize(**opt.norm_cfg),
        dtf.ToTensor()])

    # model
    model = Model(opt).to(opt.device)

    # resume
    if osp.isfile(args.chekpoint):
        print("=> loading checkpoint '{}'".format(args.chekpoint))
        checkpoint = torch.load(args.chekpoint)
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}' (epoch {})"
              .format(args.chekpoint, checkpoint['epoch']))
    else:
        raise FileNotFoundError

    labels_dir = osp.join(root_dir, 'SegmentationClass')


    model.eval()
    num = np.zeros((112,30), dtype=np.float32)#[0] 30
    full=np.zeros((112,30), dtype=np.float32)#[0] 30
    MAE=0
    MSE=0
    sumer=0
    with torch.no_grad():
        for ii, img_path in enumerate(tqdm(imgs_path)):#get test image
            
            if ii%30!=0:
                continue

            img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
            img=cv2.resize(img, (test_size[0], test_size[1] ))
            sample = {"image": img, "label": None}
            sample = transform(sample)

            # for line in open(root_dir+"/vallist.txt","r"):
            #     liner=line
            liner=img_path[-15:-10]
            label_path = osp.join(labels_dir, liner+'_'+img_path[-9:-4]+'s.hdf5')#get label.hdf5 
            print(img_path,label_path)
            with h5py.File(label_path, 'r') as hf:
                label = np.array(hf['label'])#get label.img

            label=cv2.resize(label, (mask_size[0], mask_size[1] ))

            # predict
            start=time.time()
            density_pred = model(sample['image'].unsqueeze(0).to(opt.device))
            end=time.time()
            # print(end-start)

            density_preder = torch.clamp((density_pred/balance).squeeze(0).squeeze(0), min=0.0).cpu().numpy()
            # density_preder = ((density_pred/balance).squeeze(0).squeeze(0)).cpu().numpy()
            #pred = np.clip(density_preder-noise+region_pred*noise,a_min=None,a_max=None)       #后处理优化
            pred=density_preder

            # print(density_pred.max(),density_pred.min())
            # print(region_pred.max(),region_pred.min())

            #preds = cv2.resize(np.clip(cv2.filter2D(pred,-1,fil),a_min=0.0,a_max=None)/16,(int(mask_size[0]/4),int(mask_size[1]/4)) )  #torch.clamp((density_pred/balance).squeeze(0).squeeze(0), min=0.0).cpu().numpy()

            # ans=0
            # up=1.0
            # down=0.02
            # for i in range(preds.shape[0]):
            #     for j in range(preds.shape[1]): 
            #         # if preds[i][j]>down and preds[i][j]<up:
            #         #     ans+=up-preds[i][j]

            #         if preds[i][j]>=0.1 and preds[i][j]<0.2:
            #             ans-=preds[i][j]

            #         if preds[i][j]>=0.2 and preds[i][j]<0.3:#主要误差0.3
            #             ans+=preds[i][j]*2

            #         if preds[i][j]>=0.3 and preds[i][j]<1.0:
            #             ans+=1.0-preds[i][j]

            #         if preds[i][j]>=1.0 and preds[i][j]<2.0:
            #             ans+=preds[i][j]  

            #         if preds[i][j]>=2.0 and preds[i][j]<3.0:
            #             ans+=preds[i][j]*0.3

            #         if preds[i][j]>=3.0 and preds[i][j]<4.0:
            #             ans-=preds[i][j]*0.2
            #         if preds[i][j]>=5.0:
            #             ans-=preds[i][j]*0.1

            diff=(pred.sum()*1.0)-label.sum()

            # if diff >20:
            #     print(ii)
            # print(diff,ans, ans+diff)

            # diff=ans+diff

            MAE+=np.abs(diff)
            MSE+=np.power(diff,2)

            # if(diff>10 or diff<-10):
            #     continue
            print(ii,": ",label.sum(),pred.sum(),diff)
            plt.figure()    
            plt.subplot(3, 2, 1).imshow(img)
            plt.subplot(3, 2, 3).imshow(pred)
            plt.subplot(3, 2, 4).imshow(label)
            
            # if diff <5:
            #     print ("\n",ii,"\n")

            # plt.subplot(3, 2, 6).imshow(preds)
            plt.savefig(root_dir+"/saver/"+"predict"+str(ii)+".png")
            cv2.imwrite(root_dir+"/saver/"+"original"+str(ii)+".png",img)
            # cv2.imwrite(root_dir+"/saver/"+"gt"+str(ii)+".png",label)
            # cv2.imwrite(root_dir+"/saver/"+"pre"+str(ii)+".png",density_preder)
            # plt.show()
    MSE=np.sqrt(MSE/samples)
    MAE/=samples
    print(MAE,MSE)

if __name__ == '__main__':
    test()
