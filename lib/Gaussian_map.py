

#now generate the ShanghaiA's ground truth

import os
import glob
import h5py
import scipy
import numpy as np
import scipy.spatial
import scipy.io as io
import PIL.Image as Image
from matplotlib import cm as CM
from matplotlib import pyplot as plt
from scipy.ndimage.filters import gaussian_filter 
import os.path as osp
user_dir = osp.expanduser('~')
root = user_dir+"/data/ST_CC/"

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


def main():
    part_A_train = os.path.join(root,'part_A/train_data','images')
    part_A_test = os.path.join(root,'part_A/test_data','images')
    part_B_train = os.path.join(root,'part_B/train_data','images')
    part_B_test = os.path.join(root,'part_B/test_data','images')
    mask_dir = root+ 'part_B/SegmentationClass'
    mode="val"
    if mode=="val":
        img_dir="00000"
    else:
        img_dir="00001"
    """
    path_sets = [part_A_train,part_A_test]
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)
    for img_path in img_paths:
        print(img_path)
        mat = io.loadmat(img_path.replace('.jpg', '.mat').
                         replace('images', 'ground_truth').
                         replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter_density(k)
        with h5py.File(img_path.replace('.jpg', '.h5').
                       replace('images', 'ground_truth'), 'w') as hf:
            hf['density'] = k
    # show result
    plt.imshow(Image.open(img_paths[2]))
    plt.show()
    gt_file = h5py.File(img_paths[0].
                        replace('.jpg', '.h5').
                        replace('images', 'ground_truth'), 'r')
    groundtruth = np.asarray(gt_file['density'])
    plt.imshow(groundtruth, cmap=CM.jet)
    plt.show()
    print(np.sum(groundtruth))  # don't mind this slight variation
    """
    # part_B
    if mode =="train":
        path_sets = [ part_B_test]
    else:
        path_sets = [ part_B_train]#
    img_paths = []
    for path in path_sets:
        for img_path in glob.glob(os.path.join(path, '*.jpg')):
            img_paths.append(img_path)

    for img_path in img_paths:

        mat = io.loadmat(img_path.replace('.jpg', '.mat').
                         replace('images', 'ground_truth').
                         replace('IMG_', 'GT_IMG_'))
        img = plt.imread(img_path)
        k = np.zeros((img.shape[0], img.shape[1]))
        gt = mat["image_info"][0, 0][0, 0][0]
        for i in range(0, len(gt)):
            if int(gt[i][1]) < img.shape[0] and int(gt[i][0]) < img.shape[1]:
                k[int(gt[i][1]), int(gt[i][0])] = 1
        k = gaussian_filter(k, 15)

        im=int((img_path.split("_")[-1])[:-4])
        if im+1<10:
                im="00"+str(im+1)
        elif im+1>=10 and im+1<100:
            im="0"+str(im+1)
        else:
            im=str(im+1)
        im="00"+im+"s"

        maskname = osp.join(mask_dir, img_dir+'_'+im+'.hdf5')
        with h5py.File(maskname, 'w') as hf: #img_path.replace('.jpg', '.h5').replace('images', 'density_map')
                       
            hf['density'] = k


if __name__ == '__main__':
    main()