import os
import cv2
import pickle
import h5py
import numpy as np
import os.path as osp
from PIL import Image
import sys
import os.path as osp
sys.path.insert(0, osp.join(osp.dirname(osp.abspath(__file__)), '../../'))
import torch
from torchvision import transforms
from torch.utils.data import Dataset
from dataloaders import data_process as dtf


class QNRF(Dataset):
    def __init__(self, opt, mode):
        self.data_dir = opt.root_dir_qnrf
        self.mode = mode
        self.labels_dir = osp.join(self.data_dir, 'SegmentationClass')#augment labels still here
        self.list_file = osp.join(self.data_dir, "{}list.txt".format(mode))
        self.img_root = osp.join(self.data_dir, "sequences")#sequences
        self.aug_root = osp.join(self.data_dir, "augment")#sequences
        self.anno_dir = osp.join(self.data_dir, "annotations")
        self.cache_path = self._cre_cache_path(self.data_dir)
        self.cache_file = osp.join(self.cache_path, self.mode + '_samples.pkl')
        self.img2id = dict()
        self.img_list = self._load_imgs_idx()  # order: 1,get augment/sequences images
        self.samples = self._load_samples()  # order: 2, make sample

        # transform
        if self.mode == "train":
            self.transform = transforms.Compose([
                dtf.FixedNoMaskResize(size=(1024,768)),#不调用config,A
                # dtf.RandomFilter(),
                # dtf.RandomColorJeter(0.1, 0.01, 0.01, 0),#brightness, contrast, saturation, hue
                dtf.RandomHorizontalFlip(),#
                dtf.Normalize(**opt.norm_cfg),
                dtf.ToTensor()])
        else:
            self.transform = transforms.Compose([
                dtf.FixedNoMaskResize(size=(1024,768)),  # A
                # dtf.RandomFilter(),
                dtf.Normalize(**opt.norm_cfg),
                dtf.ToTensor()])

    def __getitem__(self, index):
        sample = self.samples[index]
        img_path = sample["image"]
        dir_name, img_name = img_path.split('/')[-2:]
        label_path = osp.join(self.labels_dir, dir_name+'_'+img_name[:-4]+'s.hdf5')#get label.hdf5 
        assert osp.isfile(img_path), '{} not exist'.format(img_path)
        assert osp.isfile(label_path), '{} not exist'.format(label_path)

        img = cv2.imread(img_path)[:, :, ::-1]  # BGR2RGB
        with h5py.File(label_path, 'r') as hf:
            label = np.array(hf['label'])#get label.img

        o_h, o_w = img.shape[:2]

        sample = {"image": img, "label": label}
        # print(label,img,"!!!!!!!!!!!!!!!!!!")
        sample = self.transform(sample)

        # if self.temporal:
        #     left_img = self._getAroundFrame(img_path, -1)
        #     right_img = self._getAroundFrame(img_path, 1)
        #     lsample = {"image": left_img, "label": None}
        #     rsample = {"image": right_img, "label": None}
        #     left_img = self.transform(lsample)["image"]
        #     right_img = self.transform(rsample)["image"]
        #     sample['image'] = torch.cat((left_img, sample["image"], right_img))

        scale = torch.tensor([sample["image"].shape[1] / o_h,
                              sample["image"].shape[0] / o_w])
        sample["scale"] = scale
        sample["path"] = img_path
        return sample

    def _getAroundFrame(self, img_path, around=1):
        dir_path, img_name = img_path[:-9], img_path[-9:]
        img_id = int(img_name[:-4])
        around_file = osp.join(dir_path, str(min(max(img_id+around, 1), 30)).zfill(5)+'.jpg')
        assert osp.isfile(around_file), '{} not exist'.format(around_file)

        return cv2.imread(around_file)[:, :, ::-1]

    def _cre_cache_path(self, data_dir):
        cache_path = osp.join(data_dir, 'cache')
        if not osp.exists(cache_path):
            os.makedirs(cache_path)
        return cache_path

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

    def _load_imgs_idx(self):
        self.dir_ids = []
        img_list = []
        with open(self.list_file, 'r') as f:
            for line in f.readlines():
                self.dir_ids.append(line.strip())
        for dir in self.dir_ids:#数据集列表
            for img in os.listdir(osp.join(self.img_root, dir)):
                img_list.append(osp.join(self.img_root, dir, img))#append every image
            # if self.mode=="val":
            #     for img in os.listdir(osp.join(self.img_root, dir)):

            #         img_list.append(osp.join(self.img_root, dir, img))#append every image
            # else: 
            #     for img in os.listdir(osp.join(self.aug_root, dir)):#文件夹下的图片全部加载进来
            #         if int(img[-5])<=4:
            #                 continue
            #         img_list.append(osp.join(self.aug_root, dir, img))#append every image
        return img_list

    def _load_samples(self):
        cache_file = self.cache_file

        # # load bbox and save to cache
        # if os.path.exists(cache_file):
        #     with open(cache_file, 'rb') as fid:
        #         samples, self.img2id = pickle.load(fid)
        #     print('{} gt samples loaded from {}'.
        #           format(self.mode, cache_file))
        #     return samples

        # load information of image and save to cache
        #gts = self._load_gts()
        samples = []
        for idx, img_path in enumerate(self.img_list):
            size = Image.open(img_path).size
            img_path = img_path.replace('\\', '/')#augment root
            dir_name, img_name = img_path.split('/')[-2:]#augment image list no need to get gt
            coordinate =None#gts[dir_name+"_"+img_name]
            samples.append({
                "image": img_path,#each image
                "width": size[0],
                "height": size[1],
                "coordinate": coordinate
            })
            self.img2id[img_path] = idx

        with open(cache_file, 'wb') as fid:
            pickle.dump((samples, self.img2id), fid, pickle.HIGHEST_PROTOCOL)
        print('wrote gt samples to {}'.format(cache_file))

        return samples

    def __len__(self):
        return len(self.img_list)


def show_image(img, coordinate):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(15, 15))
    plt.imshow(img[..., ::-1])
    for local in coordinate:
        x, y = local
        plt.scatter(x, y, c='red', s=40, alpha=0.5)
    plt.show()

