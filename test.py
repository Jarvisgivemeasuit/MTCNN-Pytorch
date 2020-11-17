import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim
from torch.utils.data import DataLoader

import time
import numpy as np
from progress.bar import Bar
from apex import amp

from utils.Args import Args
from utils.utils import *
from utils.matrics import *
from net.mtcnn import *
from net.mtcnn_utils import *


class Test:
    def __init__(self, Args):
        self.args = Args
        self.validset = self.args.validset
        self.val_loader = DataLoader(self.validset, batch_size=self.args.val_batch_size,
                                       num_workers=self.args.num_workers, shuffle=False)

        self.net = torch.load('/home/grey/Documents/mtcnn_model_saving/pnet.pth')
        # self.net = PNet().cuda()
        # weights = torch.load('/home/grey/Documents/mtcnn_model_saving/pnet.pt')
        # self.net.load_state_dict(weights)

        if self.args.apex:
            self.net, self.optimizer = amp.initialize(self.net, self.optimizer, opt_level='O1')

        self.criterion_cls = nn.CrossEntropyLoss().cuda() if self.args.cuda else nn.CrossEntropyLoss()
        self.criterion_reg = nn.MSELoss(reduction='none')

        self.val_acc = Accuracy()
        self.val_pre = Precision(threshold=self.args.test_thres)
        self.val_rec = Recall(threshold=self.args.test_thres)

    def testing(self):
        starttime = time.time()
        self.val_acc.reset()
        num_val = len(self.val_loader)
        bar = Bar('testing', max=num_val)

        batch_time = AverageMeter()
        losses1 = AverageMeter()
        losses2 = AverageMeter()
        losses = AverageMeter()

        self.net.eval()
        for idx, sample in enumerate(self.val_loader):
            img, offset_label, cls_label = sample
            if self.args.cuda:
                img, offset_label, cls_label = img.cuda(), offset_label.cuda(), cls_label.cuda()

            with torch.no_grad():
                [face_cls, bbox_reg] = self.net(img)

            loss1 = self.criterion_cls(face_cls, cls_label)
            loss2 = self.criterion_reg(bbox_reg, offset_label)

            loss2 = (loss2 * cls_label.unsqueeze(1).abs()).sum().float() / cls_label.abs().sum()

            loss = loss1 + loss2
            losses1.update(loss1)
            losses2.update(loss2)
            losses.update(loss)

            self.val_acc.update(face_cls, cls_label)
            self.val_pre.update(face_cls, cls_label)
            self.val_rec.update(face_cls, cls_label)

            batch_time.update(time.time() - starttime)
            starttime = time.time()
            bar.suffix = '({batch}/{size}) Batch:{bt:.3f}s | Total:{total:} | ETA:{eta:} | Loss:{loss:.4f},loss1:{loss1:.4f},loss2:{loss2:.4f} | Acc:{Acc:.4f} | Pre:{pre:.4f} | Rec:{rec:.4f}'.format(
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

        print('test Loss: %.3f' % losses.avg)


def test():
    args = Args()
    tester = Test(args)

    print("==> Start testing")
    print('Starting Epoch:', args.start_epoch)

    tester.testing()

test()