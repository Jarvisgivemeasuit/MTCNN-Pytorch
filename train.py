import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.functional import batch_norm
import torch.optim
from torch.utils.data import DataLoader

import time
import os
from progress.bar import Bar
from apex import amp
import cProfile

from utils.Args import Args
from utils.utils import *
from utils.metrics import *
from net.mtcnn import *
from net.mtcnn_utils import *


class Trainer:
    def __init__(self, Args):
        self.args = Args
        self.trainset = self.args.trainset
        self.validset = self.args.validset
        self.train_loader = DataLoader(self.trainset, batch_size=self.args.tr_batch_size, 
                                       num_workers=self.args.num_workers, shuffle=True)
        self.val_loader = DataLoader(self.validset, batch_size=self.args.val_batch_size,
                                       num_workers=self.args.num_workers, shuffle=False)

        os.environ['CUDA_VISBLE_DEVICES'] = self.args.gpu_id

        if self.args.net == 'pnet':
            self.net = PNet().cuda() if self.args.cuda else PNet()
        elif self.args.net == 'rnet':
            self.net = RNet().cuda() if self.args.cuda else RNet()
        elif self.args.net == 'onet':
            self.net = ONet().cuda() if self.args.cuda else ONet()
        initialize_weights(self.net)

        self.optimizer = torch.optim.SGD(self.net.parameters(), lr=self.args.lr, 
                                         momentum=0.9, weight_decay=self.args.weight_decay)
        # self.scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer, [40, 70, 100], 0.5)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, 
                                                                    len(self.train_loader) * self.args.epochs, 
                                                                    1e-6)

        if self.args.apex:
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O1')
        if self.args.parallel:
            self.net = nn.DataParallel(self.net, self.args.gpu_ids)

        self.criterion_cls = nn.CrossEntropyLoss(ignore_index=-1).cuda() if self.args.cuda else nn.CrossEntropyLoss(ignore_index=-1)
        # self.criterion_cls_ohem = CrossEntropyLossWithOHEM(ohem_ratio=0.7).cuda() if self.args.cuda else CrossEntropyLossWithOHEM(ohem_ratio=0.7)
        self.criterion_reg = nn.MSELoss(reduction='none')

        self.train_acc = Accuracy()
        self.train_pre = Precision()
        self.train_rec = Recall()

        self.val_acc = Accuracy()
        self.val_pre = Precision()
        self.val_rec = Recall()

        self.best_acc = 0.0

    def training(self, epoch):
        num_train = len(self.train_loader)
        bar = Bar('Training', max=num_train)

        batch_time = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses = AverageMeter()

        self.train_acc.reset()
        self.train_pre.reset()
        self.train_rec.reset()

        starttime = time.time()
        self.net.train()

        for idx, sample in enumerate(self.train_loader):
            img, offset_label, cls_label = sample
            if self.args.cuda:
                img, offset_label, cls_label = img.cuda(), offset_label.cuda(), cls_label.cuda()

            self.optimizer.zero_grad()
            [face_cls, bbox_reg] = self.net(img)

            # if epoch > 40:
            #     loss1 = self.criterion_cls_ohem(face_cls, cls_label).float()
            # else:
            loss1 = self.criterion_cls(face_cls, cls_label).float()
            loss2 = self.criterion_reg(bbox_reg, offset_label)

            loss2 = (loss2 * cls_label.unsqueeze(1).abs()).sum().float() / cls_label.abs().sum()

            loss = loss1 + loss2
            # loss = loss2

            losses1.update(loss1)
            losses2.update(loss2)
            losses.update(loss)

            if self.args.apex:
                with amp.scale_loss(loss, self.optimizer) as scale_loss:
                    scale_loss = scale_loss.half()
                    scale_loss.backward()
            else:
                loss.backward()

            self.optimizer.step()
            self.scheduler.step()

            self.train_acc.update(face_cls, cls_label)
            self.train_pre.update(face_cls, cls_label)
            self.train_rec.update(face_cls, cls_label)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f}, loss1:{loss1:.4f}, loss2:{loss2:.4f} | Acc:{acc:.4f} | Pre:{pre:.4f} | Rec:{rec:.4f}'.format(
                batch=idx + 1,
                size=len(self.train_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss1=losses1.avg,
                loss2=losses2.avg,
                acc=self.train_acc.get(),
                pre=self.train_pre.get(),
                rec=self.train_rec.get()
            )
            bar.next()
        bar.finish()

        print('[Epoch: %d, numImages: %5d]' % (epoch, num_train * self.args.tr_batch_size))
        print('Train Loss: %.3f' % losses.avg)

    def validation(self, epoch):
        num_val = len(self.val_loader)
        bar = Bar('validation', max=num_val)

        batch_time = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses = AverageMeter()

        self.val_acc.reset()
        self.val_pre.reset()
        self.val_rec.reset()

        starttime = time.time()
        self.net.eval()

        for idx, sample in enumerate(self.val_loader):
            img, offset_label, cls_label = sample
            if self.args.cuda:
                img, offset_label, cls_label = img.cuda(), offset_label.cuda(), cls_label.cuda()

            with torch.no_grad():
                [face_cls, bbox_reg] = self.net(img)
            loss1 = self.criterion_cls(face_cls, cls_label)
            loss2 = self.criterion_reg(bbox_reg, offset_label)

            loss2 = (loss2 * cls_label.unsqueeze(1).abs()).sum() / cls_label.abs().sum()

            loss = loss1 + loss2
            losses1.update(loss1)
            losses2.update(loss2)
            losses.update(loss)

            self.val_acc.update(face_cls, cls_label)
            self.val_pre.update(face_cls, cls_label)
            self.val_rec.update(face_cls, cls_label)

            batch_time.update(time.time() - starttime)
            starttime = time.time()

            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f}, loss1:{loss1:.4f}, loss2:{loss2:.4f} | Acc:{Acc:.4f} | Pre:{pre:.4f} | Rec:{rec:.4f}'.format(
                batch=idx + 1,
                size=len(self.val_loader),
                bt=batch_time.avg,
                total=bar.elapsed_td,
                eta=bar.eta_td,
                loss=losses.avg,
                loss1=losses1.avg,
                loss2=losses2.avg,
                Acc=self.val_acc.get(),
                pre=self.val_pre.get(),
                rec=self.val_rec.get()
            )
            bar.next()
        bar.finish()

        if self.best_acc < self.val_acc.get() and self.args.save_model and epoch != 0:
            self.best_acc = self.val_acc.get()
            save_model(self.net, self.args.net, epoch, self.val_acc.get())
            print("model saving complete.")

        print('[Epoch: %d, numImages: %5d]' % (epoch, num_val * self.args.val_batch_size))
        print('validation Loss: %.3f' % losses.avg)


def train():
    args = Args()
    trainer = Trainer(args)

    print("==> Start training")
    print('Total Epoches:', args.epochs)
    print('Starting Epoch:', args.start_epoch)

    # trainer.validation(0)
    for epoch in range(args.start_epoch, args.epochs + 1):
        trainer.training(epoch)
        trainer.validation(epoch)


def test_loader():
    args = Args()
    trainset = args.trainset
    train_loader = DataLoader(trainset, batch_size=args.tr_batch_size, 
                                       num_workers=args.num_workers, shuffle=True)
    num_train = len(train_loader)
    # bar = Bar('testing', max=num_train)
    starttime = time.time()
    for idx, sample in enumerate(train_loader):
        print(time.time() - starttime)
        starttime = time.time()
    #     bar.suffix = '({batch}/{size})'.format(batch=idx + 1, size=num_train)
    #     bar.next()
    # bar.finish()


if __name__ == '__main__':
    # cProfile.run('test_loader()')
    train()
    # test_loader()