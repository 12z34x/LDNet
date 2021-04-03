import torch
import torch.nn as nn
from .utils import MyEncoder
import numpy as np


import random
class CrossEntropyLoss(object):
    def __init__(self, weight=None, ignore_index=-1):
        self.ignore_index = ignore_index
        self.weight = MyEncoder(weight)

    def __call__(self, logit, target):
        n, c, h, w = logit.size()
        device = logit.device
        # self.weight=torch.tensor([1.0,random.randint(100,800)*1.0])#随机权重

        criterion = nn.CrossEntropyLoss(weight=self.weight,
                                        ignore_index=self.ignore_index,
                                        reduction='mean').to(device)
        # print(logit.max(),target.max())
        loss = criterion(logit, target.long())#

        return loss

if __name__ == "__main__":
    torch.manual_seed(1)
    loss = CrossEntropyLoss()
    a = torch.tensor([[[[1.0,1.0],[0.0,0.0]]]])
    a2 = torch.tensor([[[1.]]])
    b = torch.tensor([[[[1.0,1.0],[0.0,0.0]]]])
    print(loss(a, b))
    pass
