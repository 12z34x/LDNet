import os
import cv2
import pickle
from tqdm import tqdm
import os.path as osp
from PIL import Image
import os,sys
import matplotlib.pyplot as plt
import random
import numpy as np
# import label_process
import h5py
down_size=8
class DroneCC(object):
    def __init__(self, data_dir, mode):
        self.data_dir = data_dir
        self.mode = mode
        #join these pathes to get the specific documents
        self.list_file = osp.join(data_dir, "{}list.txt".format(mode))
        self.img_root = osp.join(data_dir, "sequences")
        self.aug_root = osp.join(data_dir, "augment")#maybe can use augment
        self.anno_dir = osp.join(data_dir, "annotations")
        self.cache_path = self._cre_cache_path(self.data_dir)
        self.cache_file = osp.join(self.cache_path, self.mode + '_samples.pkl')
        self.img_list = self._load_imgs_idx()  # order: 1
        self.samples = self._load_samples()  # order: 2
        #cache_dir ?
    def _cre_cache_path(self, data_dir):
        cache_path = osp.join(data_dir, 'cache')
        if not osp.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path
    #get gt
    def _load_gts(self):
        gts = {}
        for dir_id in self.dir_ids:
            with open(osp.join(self.anno_dir, dir_id+'.txt')) as f:
                for line in f.readlines():
                    frame, x, y = line.split(',')
                    frame = frame.strip().zfill(5)
                    key = dir_id + '_' + frame+'.jpg'
                    if key in gts:
                        gts[key].append([int(x.strip()), int(y.strip())])
                    else:
                        gts[key] = [[int(x), int(y)]]
        return gts
    #load image sequence
    def _load_imgs_idx(self):
        self.dir_ids = []
        img_list = []
        with open(self.list_file, 'r') as f:
            for line in f.readlines():
                self.dir_ids.append(line.strip())
        for dir in self.dir_ids:
            for img in os.listdir(osp.join(self.img_root, dir)):
                img_list.append(osp.join(self.img_root, dir, img))

        return img_list
    #load samples
    def _load_samples(self):
        cache_file = self.cache_file

        # # load bbox and save to cache
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         samples, _ = pickle.load(fid)
        #     print('{} gt samples loaded from {}'.
        #           format(self.mode, cache_file))
        #     return samples

        # load information of image and save to cache
        gts = self._load_gts()#annotate this line if test set
        samples = []
        for img_path in self.img_list:
            size = Image.open(img_path).size
            img_path = img_path.replace('\\', '/')
            dir_name, img_name = img_path.split('/')[-2:]
            coordinate =gts[dir_name+"_"+img_name] #be None if test set
            samples.append({
                "image": img_path,
                "width": size[0],
                "height": size[1],
                "coordinate": coordinate
            })

        with open(cache_file, 'wb') as fid:
            pickle.dump(samples, fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt samples to {}'.format(cache_file))

        return samples

#draw the image
def show_imager(num,img, coordinate):
    plt.figure(figsize=(15, 15))
    plt.imshow(img[..., ::-1])
    for local in coordinate:
        x, y = local
        plt.scatter(x, y, c='red', s=1, alpha=0.5)
    plt.savefig('/home/twsf/data/DroneCC/augment/'+str(num)+'.jpg')
    #plt.show()

#split
def build_tiling(img_shape, split=(3, 2)):#(width,height);(width_split,height_split)
    """img_shape: w, h
    """
    stride_w = img_shape[0] / split[0]#per width
    stride_h = img_shape[1] / split[1]#per height
    shift_x = np.arange(0, split[0]) * stride_w
    shift_y = np.arange(0, split[1]) * stride_h
    shift_x, shift_y = np.meshgrid(shift_x, shift_y)
    shifts = np.vstack((
        shift_x.ravel(), shift_y.ravel(),
        shift_x.ravel()+stride_w, shift_y.ravel()+stride_h
    )).transpose().astype(np.int)

    return shifts


if __name__ == '__main__':
    # dataset = DroneCC('/home/twsf/data/DroneCC/', 'val')#process train
    # longth=len(dataset.samples)
    # width=2048
    # height=1024
    # semi_h=int(height/2)
    # semi_w=int(width/2)
    # stride=4

    # spliter=build_tiling((width,height),(stride,stride))
    # spliters=build_tiling((width/down_size,height/down_size),(stride,stride))
    # for i in range(0,int(longth/30)):#use double circulation
    #     spl_image=[]
    #     spl_mask=[]
    #     aug_f=(dataset.samples[i*30])['image'].split('/')[-2]#get index
    #     mask_dir =  '/home/twsf/data/DroneCC/augment/'+ aug_f
    #     if not osp.exists(mask_dir):
    #         os.mkdir(mask_dir)

    #     im=0
    #     for j in range(30):
    #         sample = dataset.samples[30*i+j]#split images in per video
    #         density_mask= label_process._generate_mask(sample)#get mask of each image
    #         img = cv2.imread(sample['image'])
    #         img=cv2.resize(img,(width,height))#width and height
    #         for k in range(stride*stride):
    #             spl_image.append(img[spliter[k][1]:spliter[k][3],spliter[k][0]:spliter[k][2]])
    #             spl_mask.append(density_mask[spliters[k][1]:spliters[k][3],spliters[k][0]:spliters[k][2]])
    #     #print("Split "+str(i)+ "-th images and masks!")
    #     state=random.getstate()
    #     random.shuffle(spl_image)
    #     random.setstate(state)
    #     random.shuffle(spl_mask)#shuffle mask in the same order
    #     #print("Shuffle "+str(i)+ "-th images and masks!")
    #     photor=[]
    #     denser=[]
    #     for ii in range (30*stride*2):
    #         photo=spl_image[int(ii*stride/2)]
    #         dense=spl_mask[int(ii*stride/2)]
    #         for jj in range(1,int(stride/2)):
    #             photo=np.hstack((photo,spl_image[int(jj+ii*stride/2)]))
    #             dense=np.hstack((dense,spl_mask[int(jj+ii*stride/2)]))
    #         photor.append(photo)
    #         denser.append(dense)
    #     imager=[]
    #     density=[]
    #     for ii in range (30*4):
    #         if ii%4==0:
    #             im+=1
    #         imager=photor[int(ii*stride/2)]
    #         density=denser[int(ii*stride/2)]
    #         for jj in range(1,int(stride/2)):
    #             imager=np.vstack((imager,photor[int(jj+ii*stride/2)]))
    #             density=np.vstack((density,denser[int(jj+ii*stride/2)]))
    #         if im<10:
    #             fold='0000'+str(im)
    #         else: 
    #             fold='000'+str(im)
    #         for adder in range(stride):
    #             cv2.imwrite('/home/twsf/data/DroneCC/augment/'+aug_f+'/'+fold+'-'+str(adder+1)+'.jpg',imager)
    #             maskname = osp.join('/home/twsf/data/DroneCC/SegmentationClass', aug_f+'_'+fold+'-'+str(adder+1)+'.hdf5')
    #             with h5py.File(maskname, 'w') as hf: 
    #                  hf['label'] = density
    #             # label_process.show_image(imager,density,adder)
    #     print("Complete "+str(i)+ "-th images and masks!")

    # for i in range(0,int(longth)):#use double circulation
    #     counter=4#Very important!
    #     sample = dataset.samples[i]#split images in per video
    #     img_path = sample['image']
    #     img = cv2.imread(img_path)
    #     aug_f=img_path.split('/')[-2]#get index
    #     aug_i=img_path.split('/')[-1]
    #     img=cv2.resize(img,(width,height))#width and height
    #     img1=img[0:semi_h,0:semi_w]#height,width
    #     img2=img[semi_h:height,0:semi_w]
    #     img3=img[0:semi_h,semi_w:width]
    #     img4=img[semi_h:height,semi_w:width]
    #     for j in (img1,img2,img3,img4):
    #         counter+=1
    #         cv2.imwrite('/home/twsf/data/DroneCC/augment/'+aug_f+'/'+aug_i.split('.')[0]+'-'+str(counter)+'.jpg',j)
         # coordinate = sample['coordinate']
         # show_imager(i,img, coordinate)#add index
        pass
