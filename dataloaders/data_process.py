import cv2
import random
import numpy as np
from PIL import Image, ImageFilter
from torchvision.transforms import ColorJitter
import torch


#rotate is no use
# class RandomRotate(object):
#     def __init__(self, degree):
#         self.degree = degree

#     def __call__(self, sample):
#         img = sample['image']
#         mask = sample['label']
#         rotate_degree = random.uniform(-1*self.degree, self.degree)
#         img = img.rotate(rotate_degree, Image.BILINEAR)
#         mask = mask.rotate(rotate_degree, Image.NEAREST)

#         return {'image': img,
#                 'label': mask}



#normalize the image and para multiply label
class Normalize(object):
    """Normalize a tensor image with mean and standard deviation.
    Args:
        mean (tuple): means for each channel.
        std (tuple): standard deviations for each channel.
    """
    def __init__(self,
                 mean=(0.485, 0.456, 0.406),
                 std=(0.229, 0.224, 0.225),
                 para=1,
                 maximum=None):
        assert max(mean) <= 1 and max(std) <= 1, "mean or std value error!"
        self.mean = mean
        self.maximum = maximum
        self.std = std
        self.para = para

    def __call__(self, sample):
        sample['image'] = sample['image'].astype(np.float32)
        if sample['label'] is not None:
            sample['label'] = (sample['label'] * self.para).astype(np.float32)
            if self.maximum is not None:
                sample['label'] = torch.clamp(sample['label'], max=self.maximum)

        sample['image'] /= 255.0
        sample['image'] -= self.mean
        sample['image'] /= self.std

        return sample

class RandomColorJeter(object):
    def __init__(self, brightness=0, contrast=0, saturation=0, hue=0):
        self.tr = ColorJitter(brightness, contrast, saturation, hue)

    def __call__(self, sample):
        sample['image'] = self.tr(Image.fromarray(sample['image']))
        sample['image'] = np.array(sample['image'])

        return sample

class RandomGaussianBlur(object):
    def __call__(self, sample):
        img = Image.fromarray(sample['image'])
        if random.random() < 0.5:
            img = img.filter(ImageFilter.GaussianBlur(
                radius=random.random()))
            sample['image'] = np.array(sample['image'])

        return sample

class RandomFilter(object):
    def __init__(self):
        self.kernel = np.array([[0, -1, 0],
                                [-1, 5, -1],
                                [0, -1, 0]])

    def __call__(self, sample):
        if random.random() < 0.5:
            sample['image'] = cv2.filter2D(sample['image'], -1, self.kernel)

        return sample

#flip image and label
class RandomHorizontalFlip(object):
    def __call__(self, sample):
        if random.random() < 0.5:
            sample['image'] = sample['image'][:, ::-1, :]
            if sample['label'] is not None:
                sample['label'] = sample['label'][:, ::-1]

        return sample

#resize the input
class FixedNoMaskResize(object):
    def __init__(self, size):
        if isinstance(size, int):
            self.size = (size, size)
        elif isinstance(size, tuple):
            self.size = size  # size: (w, h)

    def __call__(self, sample):
        sample['image'] = cv2.resize(sample['image'], self.size  )

        return sample

#transform to tensor
class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        sample['image'] = torch.from_numpy(sample['image']).permute(2, 0, 1)#将第三维度——通道提至最前
        if sample['label'] is not None:
            sample['label'] = torch.from_numpy(sample['label'])

        return sample

if __name__ == "__main__":
    from torchvision import transforms
    img = cv2.imread("/home/twsf/work/EvaNet/data/Visdrone_Region/JPEGImages/0000001_02999_d_0000005.jpg")
    gt = cv2.imread("/home/twsf/work/EvaNet/data/Visdrone_Region/SegmentationClass/0000001_02999_d_0000005.png")
    pair = {'image': img, 'label': gt}
    model = transforms.Compose([
            FixedNoMaskResize(size=(640, 480)),
            RandomColorJeter(0.3, 0.3, 0.3, 0.3),
            RandomHorizontalFlip(),
            Normalize(),
            ToTensor()])
    sample = model(pair)
    pass
