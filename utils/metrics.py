import torch
import numpy as np


class AverageMeter:
    def __init__(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def reset(self):
        self.count = 0
        self.sum = 0
        self.avg = 0
        self.val = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class Accuracy:
    def __init__(self, ignore_idx=-1, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0
        self.ignore_idx = ignore_idx
        self.eps = eps

    def update(self, pred, target):
        ignore_mask = target != self.ignore_idx
        pred = torch.argmax(pred, dim=1)

        self.num_correct += ((pred == target) * ignore_mask).sum().item()
        self.num_instance += ignore_mask.sum().item()

    def get(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_instance = 0
        self.num_correct = 0


class Precision:
    def __init__(self, ignore_idx=-1, threshold=0.5, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0
        self.ignore_idx = ignore_idx
        self.threshold = threshold
        self.eps = eps

    def update(self, pred, target):
        ignore_mask = target != self.ignore_idx
        pred = torch.softmax(pred, dim=1)
        # print('  ', pred[:, 1])
        pred = pred[:, 1] > self.threshold
        # print(' ', (pred * target * ignore_mask).sum().item(), (target * ignore_mask).sum().item())

        self.num_correct += (pred * target * ignore_mask).sum().item()
        self.num_instance += (pred * ignore_mask).sum().item()

    def get(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0


class Recall:
    def __init__(self, ignore_idx=-1, threshold=0.5, eps=1e-7):
        self.num_correct = 0
        self.num_instance = 0
        self.ignore_idx = ignore_idx
        self.threshold = threshold
        self.eps = eps

    def update(self, pred, target):
        ignore_mask = target != self.ignore_idx
        pred = torch.softmax(pred, dim=1)
        pred = pred[:, 1] > self.threshold

        self.num_correct += (pred * target * ignore_mask).sum().item()
        self.num_instance += (target * ignore_mask).sum().item()

    def get(self):
        return self.num_correct / (self.num_instance + self.eps)

    def reset(self):
        self.num_correct = 0
        self.num_instance = 0
