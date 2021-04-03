"""convert VOC format
+ density_voc
    + JPEGImages
    + SegmentationClass
"""
import os
import cv2
import h5py
import argparse
import numpy as np
import scipy.spatial
import os.path as osp
from tqdm import tqdm
from scipy.ndimage.filters import gaussian_filter
from scipy.spatial.distance import pdist
from dataset import DroneCC
import matplotlib.pyplot as plt
import matplotlib.cm as cm
user_dir = osp.expanduser('~')
W=2048
H=1024#qnrf
down_size=8
semi_h=int(H/(2*down_size))
semi_w=int(W/(2*down_size))



#analyse args
def parse_args():
    parser = argparse.ArgumentParser(description="convert to voc dataset")
    parser.add_argument('--mode', type=str, default=['val'], #choose the data_set
                        nargs='+', help='for train and val')
    parser.add_argument('--db_root', type=str,
                        default=user_dir+"/data/UCF_QNRF/",#change dataset!!!
                        # default="G:\\CV\\Dataset\\CC\\Visdrone\\VisDrone2020-CC",
                        help="dataset's root path")
    parser.add_argument('--method', type=str, default='default',
                        choices=['centerness', 'gauss', 'default'])
    parser.add_argument('--maximum', type=int, default=999,
                        help="maximum of mask")
    parser.add_argument('--show', type=bool, default=False,
                        help="show image and region mask")
    args = parser.parse_args()

    return args

def gaussian_filter_density(gt):
    print(gt.shape)
    density = np.zeros(gt.shape, dtype=np.float32)
    gt_count = np.count_nonzero(gt)
    if gt_count == 0:
        return density

    pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
    leafsize = 2048
    # build kdtree
    tree = scipy.spatial.KDTree(pts.copy(), leafsize=leafsize)
    # query kdtree
    distances, locations = tree.query(pts, k=4)

    print('generate density...')
    for i, pt in enumerate(pts):
        pt2d = np.zeros(gt.shape, dtype=np.float32)
        pt2d[pt[1], pt[0]] = 1.0
        if gt_count > 1:
            sigma = (distances[i][1] + distances[i][2] + distances[i][3]) * 0.1
        else:
            sigma = np.average(np.array(gt.shape)) / 2.0 / 2.0

        density += gaussian_filter(pt2d, sigma, mode="constant")

    print('done.')
    return density

#draw image
def show_image(img, m,l):
    args = parse_args()
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 2, 1).imshow(img)
    plt.subplot(2, 2, 2).imshow(m, cmap=cm.jet)
    plt.savefig(args.db_root+'mask_image'+str(l)+'.jpg')
    #plt.show()

#draw image
def show_image_aug(img, m1,m2,m3,m4,l):
    args = parse_args()
    plt.figure(figsize=(10, 10))
    plt.subplot(2, 3, 4).imshow(img)
    plt.subplot(2, 3, 2).imshow(m1, cmap=cm.jet)
    plt.subplot(2, 3, 3).imshow(m2, cmap=cm.jet)
    plt.subplot(2, 3, 5).imshow(m3, cmap=cm.jet)
    plt.subplot(2, 3, 6).imshow(m4, cmap=cm.jet)
    plt.savefig(args.db_root+str(l)+'aug_image.jpg')
    #plt.show()

#gaussian filter
def gaussian_pattern(height,width):
    """在3倍的gamma距离内的和高斯的97%
    """
    cx = int(round((width-0.01)/2))
    cy = int(round((height-0.01)/2))
    pattern = np.zeros((width, height), dtype=np.float32)
    pattern[cy, cx] = 1  #the value of center point is 1
    gamma = [0.15*width, 0.15*height]#gaussian standard deviation
    pattern = gaussian_filter(pattern, gamma)#gausssian filtering

    return pattern

#sample is an image
def _generate_mask(sample,down_size=8):
    try:
        #size of image is different
        width,height  = sample["width"],sample["height"]
        #print(sample["coordinate"].len)
        #size of mask

        mask_w,mask_h  =int((width*W)/(down_size*1920)), int((height*H)/(down_size*1080)) #mask_scale 
        density_mask = np.zeros((mask_h, mask_w), dtype=np.float32)#[0] height;[1] width 
        #find the nearest point; then calcuate the min-distance;gaussian padding
        ans=np.zeros((width*height), dtype=np.uint8)#space exchange time
        miner=1920*1080
        for i, pointA in enumerate(sample["coordinate"]):#enumerate as the point sequence
            if miner<=4:
                break
            for j,pointB in enumerate(sample["coordinate"]):
                if i==j or ans[j]==1:
                    continue
                P=np.vstack([pointA,pointB])
                distance=pdist(P,'seuclidean')
                if distance<miner:
                    miner=distance

            ans[i]=1
        kernal_size=int(max((miner[0]*W)/(down_size*width),1))#常为1
        gaussian_kernal=gaussian_pattern(kernal_size, kernal_size)
        for k, pointC in enumerate(sample["coordinate"]):
            px=int((pointC[0]*W)/(down_size*width))
            py=int((pointC[1]*H)/(down_size*height))
            #get pointA's position in mask
            for m in range(kernal_size ):#protect the border
                for n in range(kernal_size):
                    if py-kernal_size+m<0 or py+kernal_size+m>mask_h or px-kernal_size+n<0 or px+kernal_size+n>mask_w:
                        continue   
                    density_mask[py-kernal_size+m,px-kernal_size+n]+= gaussian_kernal[m,n]
        return density_mask

    except Exception as e:
        print(e)
        print(sample["image"])

def _generate_mask_UCF(sample,down_size=8):
    try:
        #size of image is different
        W=1024
        H=512
        width,height  = sample["width"],sample["height"]
        #print(sample["coordinate"].len)
        #size of mask

        mask_w,mask_h  =int(W/down_size), int(H/down_size)#mask_scale 

        density_mask = np.zeros((mask_h, mask_w), dtype=np.float32)#[0] height;[1] width 
        #find the nearest point; then calcuate the min-distance;gaussian padding
        ans=np.zeros((width*height), dtype=np.uint8)#space exchange time
        miner=1024*512
        for i, pointA in enumerate(sample["coordinate"]):#enumerate as the point sequence
            if miner <=8:
                break
            for j,pointB in enumerate(sample["coordinate"]):
                if i==j or ans[j]==1:
                    continue
                P=np.vstack([pointA,pointB])
                distance=pdist(P,'seuclidean')
                if distance<miner:
                    miner=distance

            ans[i]=1
        kernal_size=int(max((miner[0]*W)/(down_size*width),1))
        gaussian_kernal=gaussian_pattern(kernal_size, kernal_size)
        for k, pointC in enumerate(sample["coordinate"]):
            px=int((pointC[0]*W)/(down_size*width))
            py=int((pointC[1]*H)/(down_size*height))
            #get pointA's position in mask
            for m in range(kernal_size ):#protect the border
                for n in range(kernal_size):
                    if py-kernal_size+m<0 or py+kernal_size+m>mask_h or px-kernal_size+n<0 or px+kernal_size+n>mask_w:
                        continue
                    density_mask[py-kernal_size+m,px-kernal_size+n]+= gaussian_kernal[m,n]
        return density_mask

    except Exception as e:
        print(e)
        print(sample["image"])

def _generate_mask_ST(sample,down_size=8):
    try:
        #size of image is different
        W=1024
        H=512
        width,height  = sample["width"],sample["height"]
        #print(sample["coordinate"].len)
        #size of mask

        mask_w,mask_h  =int(W/down_size), int(H/down_size)#mask_scale 
        density_mask = np.zeros((mask_h, mask_w), dtype=np.float32)#[0] height;[1] width 
        #find the nearest point; then calcuate the min-distance;gaussian padding
        ans=np.zeros((width*height), dtype=np.uint8)#space exchange time
        miner=1024*768
        for i, pointA in enumerate(sample["coordinate"]):#enumerate as the point sequence
            if miner <=8:#有2倍关系
                break
            for j,pointB in enumerate(sample["coordinate"]):
                if i==j or ans[j]==1:
                    continue
                P=np.vstack([pointA,pointB])
                distance=pdist(P,'seuclidean')
                if distance<miner:
                    miner=distance

            ans[i]=1
        kernal_size=int(max((miner[0]*W)/(down_size*width),1))
        gaussian_kernal=gaussian_pattern(kernal_size, kernal_size)
        for k, pointC in enumerate(sample["coordinate"]):

            px=int((pointC[0]*W)/(down_size*width))
            py=int((pointC[1]*H)/(down_size*height))
            #get pointA's position in mask
            for m in range(kernal_size ):#protect the border
                for n in range(kernal_size):
                    if py-kernal_size+m<0 or py+kernal_size+m>mask_h or px-kernal_size+n<0 or px+kernal_size+n>mask_w:
                        # density_mask[py-kernal_size+m,px-kernal_size+n]+= gaussian_kernal[m,n]
                        continue   
                    density_mask[py-kernal_size+m,px-kernal_size+n]+= gaussian_kernal[m,n]
        # for i in range(mask_h):
        #     for j in range(mask_w):
        #         if density_mask[i][j]>0:
        #             continue
        #         density_mask[i][j]=-1.0
        # print(density_mask.max())
        return density_mask

    except Exception as e:
        print(e)
        print(sample["image"])

def _generate_mask_STer(sample,down_size=8):
    # try:
    #size of image is different
    W=1024
    H=768
    width,height  = sample["width"],sample["height"]
    #print(sample["coordinate"].len)
    #size of mask

    mask_w,mask_h  =int(W/down_size), int(H/down_size)#mask_scale 
    density_mask = np.zeros((mask_h, mask_w), dtype=np.float32)#[0] height;[1] width 
    #find the nearest point; then calcuate the min-distance;gaussian padding
    kernal_size=2
    for k, pointC in enumerate(sample["coordinate"]):
        # print(pointC)
        px=int((pointC[0]*W)/(down_size*width))
        py=int((pointC[1]*H)/(down_size*height))
        if py<0 or py>=mask_h or px<0 or px>=mask_w:
            continue 
        density_mask[py,px]+= 1
    density_mask=gaussian_filter(density_mask,kernal_size)
    #density_mask=gaussian_filter_density(density_mask)
    # print(density_mask)
    return density_mask

def _generate_mask_STs(sample,down_size=8):
    # try:
    #size of image is different
    W=1024
    H=512
    width,height  = sample["width"],sample["height"]
    #print(sample["coordinate"].len)
    #size of mask

    mask_w,mask_h  =int((W)/down_size), int((H)/down_size)#mask_scale 
    density_mask = np.zeros((mask_h, mask_w), dtype=np.float32)#[0] height;[1] width 
    #find the nearest point; then calcuate the min-distance;gaussian padding
    for k, pointC in enumerate(sample["coordinate"]):
        # print(pointC)
        px=int((pointC[0]*W)/(down_size*width))
        py=int((pointC[1]*H)/(down_size*height))
        if py<0 or py>=mask_h or px<0 or px>=mask_w:
            continue 
        density_mask[py,px]+= 1
    density_mask=gaussian_filter_density(density_mask)
    # print(density_mask)
    return density_mask

    # except Exception as e:
    #     print(e)
    #     print(sample["image"])

def _generate_mask_UCFqnrf(sample,down_size=8):
#    try:
    #size of image is different
    W=1024
    H=768
    width,height  = sample["width"],sample["height"]
    #print(sample["coordinate"].len)
    #size of mask

    mask_w,mask_h  =int(W/down_size), int(H/down_size)#mask_scale 

    density_mask = np.zeros((mask_h, mask_w), dtype=np.float32)#[0] height;[1] width 
    #find the nearest point; then calcuate the min-distance;gaussian padding
    ans=np.zeros((width*height), dtype=np.uint8)#space exchange time
    miner=1024*512
    for i, pointA in enumerate(sample["coordinate"]):#enumerate as the point sequence
        if miner <=8:
            break
        for j,pointB in enumerate(sample["coordinate"]):
            if i==j or ans[j]==1:
                continue
            P=np.vstack([pointA,pointB])
            distance=pdist(P,'seuclidean')
            if distance<miner:
                miner=distance

        ans[i]=1
    kernal_size=int(max((miner[0]*W)/(down_size*width),1))
    gaussian_kernal=gaussian_pattern(kernal_size, kernal_size)
    for k, pointC in enumerate(sample["coordinate"]):
        px=int((pointC[0]*W)/(down_size*width))
        py=int((pointC[1]*H)/(down_size*height))
        #get pointA's position in mask
        for m in range(kernal_size ):#protect the border
            for n in range(kernal_size):
                if py-kernal_size+m<0 or py+kernal_size+m>mask_h or px-kernal_size+n<0 or px+kernal_size+n>mask_w:
                    continue
                density_mask[py-kernal_size+m,px-kernal_size+n]+= gaussian_kernal[m,n]
    return density_mask

    # except Exception as e:
    #     print(e)
    #     print(sample["image"])

if __name__ == "__main__":
    args = parse_args()
    sumer=0
    for split in args.mode:
        dataset = DroneCC(args.db_root, split)
        mask_dir = dataset.data_dir + 'SegmentationClass'
        if not osp.exists(mask_dir):
            os.mkdir(mask_dir)
            mask_dir = dataset.data_dir + 'SegmentationClass'
        samples = dataset.samples

        #print('generate {} masks...'.format(split))
        adder=0
        for sample in tqdm(samples):
                        
            img_dir, img_name = sample['image'].split('/')[-2:]
            # print(img_dir,img_name)
            density_mask= _generate_mask_STs(sample)

            # density_mask= _generate_mask(sample)
            num=4
            # semi_h=int(sample["height"]/down_size)
            # semi_w=int(sample["width"]/down_size)

            # dense1=density_mask[0:semi_h,0:semi_w]#height,width
            # dense2=density_mask[semi_h:2*semi_h,0:semi_w]
            # dense3=density_mask[0:semi_h,semi_w:2*semi_w]
            # dense4=density_mask[semi_h:2*semi_h,semi_w:2*semi_w]

            # for dense in(dense1,dense2,dense3,dense4):
            #     num+=1#dense index
            #     maskname = osp.join(mask_dir, img_dir+'_'+img_name[:-4]+'-'+str(num)+'.hdf5')
            #     with h5py.File(maskname, 'w') as hf: 
            #         hf['label'] = dense

            # dense5=cv2.resize(density_mask, (semi_w,semi_h))
            dense5=density_mask
            sumer+=dense5.sum()
            print(dense5.sum())
            maskname = osp.join(mask_dir, img_dir+'_'+img_name[:-4]+'.hdf5')
            print("mask:",mask_dir,img_dir,img_name[:-4])
            with h5py.File(maskname, 'w') as hf: 
                hf['label'] = dense5
            adder+=1

            if adder>5:
                continue
            img = cv2.imread(sample['image'])#[0] is height;[1] is width
            show_image(img, dense5,adder)  
            # show_image_aug(img, dense1,dense3,dense2,dense4,adder)
        print(sumer)    
        print('done.')
