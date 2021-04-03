import os
import fire
import time
import collections
import numpy as np
import os.path as osp
from tqdm import tqdm
import cv2
from configs.config import opt
import matplotlib.pyplot as plt
# from models import CSMNet as Model
from models import AblaNet as Model
from models.losses import build_loss

from models.utils import Evaluator

from utils import (Saver, Timer, TensorboardSummary,
                   calculate_weigths_labels)
import torch
import torch.optim as optim
import multiprocessing #multi_threading
import random
from dataloaders.datasets import st#B

from torch.utils.data import DataLoader

multiprocessing.set_start_method('spawn', True)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)


def make_data_loader(opt, mode="train"):
    
    dataset = st.ST(opt, mode)
    dataloader = DataLoader(dataset,
                            batch_size=opt.batch_size,
                            num_workers=opt.workers,
                            shuffle=True if mode == "train" else False,
                            pin_memory=True)

    return dataset, dataloader

class Trainer(object):
    def __init__(self, mode):
        #freeze
        # super(ResNet, self).train(mode)
        # self._freeze_stages()
        # if mode and self.norm_eval:
        #     for m in self.modules():
        #         # trick: eval have effect on BatchNorm only
        #         if isinstance(m, _BatchNorm):
        #             m.eval()
        
        # Define Saver
        self.saver = Saver(opt, mode)
        self.logger = self.saver.logger

        # visualize
        self.summary = TensorboardSummary(self.saver.experiment_dir)
        self.writer = self.summary.create_summary()

        # Dataset dataloader
        self.train_dataset, self.train_loader = make_data_loader(opt)
        self.nbatch_train = len(self.train_loader)
        self.val_dataset, self.val_loader = make_data_loader(opt, mode="val")
        self.nbatch_val = len(self.val_loader)

        # model
        model = Model(opt)
        self.model = model.to(opt.device)

        # Loss
        #self.loss_density = build_loss(opt.loss_density)
        if opt.use_balanced_weights:
            classes_weights_file = osp.join(opt.root_dir, 'train_classes_weights.npy')
            if os.path.isfile(classes_weights_file):
                weight = np.load(classes_weights_file)
            else:
                weight = calculate_weigths_labels(
                    self.train_loader, opt.root_dir)

            opt.loss_region['weight'] = weight
        self.loss_region = build_loss(opt.loss_region)
        self.loss_density = build_loss(opt.loss_density)

        # Define Evaluator
        self.evaluator = Evaluator(dataset=opt.dataset)  # class_num is 2

        # Resuming Checkpoint
        self.best_pred = float('inf')
        self.best_preder = float('inf')
        self.start_epoch = 0
        if opt.resume:
            if os.path.isfile(opt.pre):
                print("=> loading checkpoint '{}'".format(opt.pre))
                checkpoint = torch.load(opt.pre)
                self.start_epoch = checkpoint['epoch']
                self.best_pred = checkpoint['best_pred']
                self.model.load_state_dict(checkpoint['state_dict'])
                print("=> loaded checkpoint '{}' (epoch {})"
                      .format(opt.pre, checkpoint['epoch']))
            else:
                print("=> no checkpoint found at '{}'".format(opt.pre))

        if len(opt.gpu_id) > 1:
            self.logger.info("Using multiple gpu")
            self.model = torch.nn.DataParallel(self.model,
                                               device_ids=opt.gpu_id)

        # Define Optimizer and Lr Scheduler
        self.optimizer = torch.optim.SGD(self.model.parameters(),
                                         lr=opt.lr,
                                         momentum=opt.momentum,
                                         #betas=opt.betas,
                                         weight_decay=opt.decay)
        # self.optimizer = torch.optim.Adam(self.model.parameters(),
        #                                  lr=opt.lr,
        #                                  betas=opt.betas,
        #                                  weight_decay=opt.decay)               
        self.scheduler = optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[8,16,32,64,128,256,512,1024,2048],#round(opt.epochs * x) for x in opt.steps], 在[12,24,36,48,60,72]
            gamma=opt.gamma)

        # Time
        self.loss_hist = collections.deque(maxlen=500)
        self.timer = Timer(opt.epochs, self.nbatch_train, self.nbatch_val)
        self.step_time = collections.deque(maxlen=opt.print_freq)

    def train(self, epoch):
        self.model.train()
        if opt.freeze_bn:
            self.model.module.freeze_bn() if len(opt.gpu_id) > 1 \
                else self.model.freeze_bn()
        last_time = time.time()
        epoch_loss = []
        for iter_num, sample in enumerate(self.train_loader):
            # if iter_num >= 2: break

            imgs = sample["image"].to(opt.device)#[0] batch;[1] channels;[3] height;[4] width

            # plt.figure()    
            # plt.subplot(1, 1, 1).imshow(np.array((imgs.cpu())[0].permute(1, 2, 0)))
            # plt.savefig("/home/twsf/data/ST_CC/part_B/saver/"+"predicter"+str(iter_num)+".png")
            density_gt = sample["label"].to(opt.device)#[0] batch;[1] height;[2] width 
            region_gt = ((sample["label"] > 0).float()).to(opt.device)
            
            # #使用focalloss
            # region_gter=((sample["label"] == 0).float()).to(opt.device)
            # region_gt=torch.cat((region_gt.repeat(1,1,1,1).transpose(0,1),region_gter.repeat(1,1,1,1).transpose(0,1)),1)

            region_pred, density_pred = self.model(imgs)

            density_pred = density_pred.clamp(min=0.0)

            region_loss = self.loss_region(region_pred, region_gt)
            density_loss = self.loss_density(density_pred, density_gt)
            loss = region_loss+ density_loss
            loss.backward()
            # torch.nn.utils.clip_grad_norm_(self.model.parameters(), 10)
            self.loss_hist.append(float(loss))
            epoch_loss.append(float(loss.cpu().item()))

            self.optimizer.step()
            self.optimizer.zero_grad()
            # self.scheduler(self.optimizer, iter_num, epoch)

            # Visualize
            global_step = iter_num + self.nbatch_train * epoch + 1
            self.writer.add_scalar('train/loss', loss.cpu().item(), global_step) 
            batch_time = time.time() - last_time
            last_time = time.time()
            eta = self.timer.eta(global_step, batch_time)
            self.step_time.append(batch_time)
            if global_step % opt.print_freq == 0:
                printline = ('Epoch: [{}][{}/{}] '
                                 'lr: {:1.7f}, '  # 10x:{:1.5f}), '
                                 'eta: {}, time: {:1.1f}, '
                                 'density loss: {:1.7f}, '
                                 'region loss:{:1.7f},'
                                 'loss: {:1.7f}').format(
                                    epoch, iter_num+1, self.nbatch_train,
                                    self.optimizer.param_groups[0]['lr'],
                                    # self.optimizer.param_groups[1]['lr'],
                                    eta, np.sum(self.step_time),
                                    density_loss,
                                    region_loss,
                                    np.mean(self.loss_hist))
                self.logger.info(printline)

            del loss, density_loss        
            # try:
            # except Exception as e:
            #     print(e)
            #     continue

        self.scheduler.step()

    def validate(self, epoch):
        start=time.time()
        self.model.eval()
        self.evaluator.reset()
        SMAE = 0
        SMSE=0
        sumer=0
        with torch.no_grad():
            tbar = tqdm(self.val_loader, desc='\r')
            for i, sample in enumerate(tbar):  

                # if i > 3: break
                imgs = sample['image'].to(opt.device)
                density_gt = sample["label"].to(opt.device)
   
                region_gt = (sample["label"] > 0).float()
                path = sample["path"]


                region_pred, density_pred = self.model(imgs)

                # metrics
                target = region_gt.numpy()
                pred = region_pred.data.cpu().numpy()
                pred = np.argmax(pred, axis=1).reshape(target.shape)
                

                self.evaluator.add_batch(target, pred, path)
                density_pred = density_pred.clamp(min=0.0)

                # plt.figure()    
                # plt.subplot(1, 1, 1).imshow(np.array((density_pred.cpu())[0].squeeze()))
                # plt.savefig("/home/twsf/data/ST_CC/part_B/saver/"+"predicter"+str(i)+".png")

                sumer+=density_gt.sum()

                for j in range(density_gt.shape[0]):
                    SMAE+=(density_gt[j].sum()-density_pred[j].sum()).abs()
                    SMSE+=((density_gt[j].sum() - density_pred[j].sum())/ opt.norm_cfg['para']).square()

            # Fast test during the training

            MAE = SMAE.cpu() / (len(self.val_dataset) * opt.norm_cfg['para'])

            MSE = np.sqrt(SMSE.cpu() / (len(self.val_dataset) ))
            self.writer.add_scalar('val/MAE', MAE, epoch)

            printline = ("Val => MAE: {:.4f}]").format(MAE)
            printliner = ("Val => MSE: {:.4f}]").format(MSE)
            self.logger.info(printline)
            self.logger.info(printliner)
        end=time.time()

        return MAE,MSE


def train(**kwargs):
    start_time = time.time()
    opt._parse(kwargs)#load and print args
    trainer = Trainer(mode="train")

    trainer.logger.info('Num training images: {}'.format(len(trainer.train_dataset)))

    for epoch in range(trainer.start_epoch, opt.epochs):
        # train
        trainer.train(epoch)

        # val
        val_time = time.time()
        pred ,preder= trainer.validate(epoch)
        trainer.timer.set_val_eta(epoch, time.time() - val_time)

        trainer.logger.info("Val[New pred: {:1.4f}, previous best: {:1.4f}]".format(
            pred, trainer.best_pred
        ))
        trainer.logger.info("Val[New pred: {:1.4f}, previous best: {:1.4f}]".format(
            preder, trainer.best_preder
        ))
        is_best = pred < trainer.best_pred
        trainer.best_pred = min(pred, trainer.best_pred)
        trainer.best_preder=min(preder,trainer.best_preder)
        if (epoch % 20 == 0 and epoch != 0) or is_best:
            trainer.saver.save_checkpoint({
                'epoch': epoch,
                'state_dict': trainer.model.module.state_dict() if len(opt.gpu_id) > 1
                else trainer.model.state_dict(),
                'best_pred': trainer.best_pred,
                'best_preder': trainer.best_preder,
                'optimizer': trainer.optimizer.state_dict(),
            }, is_best)

    all_time = trainer.timer.second2hour(time.time() - start_time)
    trainer.logger.info("Train done!, Sum time: {}, Best result: {}".format(all_time, trainer.best_pred))

    # cache result
    print("Backup result...")
    trainer.saver.backup_result()
    print("Done!")

if __name__ == '__main__':
    #train()
    fire.Fire(train)
