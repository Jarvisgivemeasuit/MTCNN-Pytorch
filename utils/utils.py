import numpy as np
import os
import cv2
import torch
from torch import nn


def make_sure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)
    return path


def iou(bbox1, bbox2):
    inter_x1 = np.maximum(bbox1[:, 0], bbox2[:, 0])
    inter_y1 = np.maximum(bbox1[:, 1], bbox2[:, 1])
    inter_x2 = np.minimum(bbox1[:, 0] + bbox1[:, 2], bbox2[:, 0] + bbox2[:, 2])
    inter_y2 = np.minimum(bbox1[:, 1] + bbox1[:, 3], bbox2[:, 1] + bbox2[:, 3])
    inter = np.maximum(inter_x2 - inter_x1, 0) * np.maximum(inter_y2 - inter_y1, 0)

    return inter / (bbox1[:, 2] * bbox1[:, 3] + bbox2[:, 2] * bbox2[:, 3] - inter)


def nms(dets, thresh):
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 0] + dets[:, 2]
    y2 = dets[:, 1] + dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort().ravel()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)

        xx1 = np.maximum(x1[i], x1[order])
        yy1 = np.maximum(y1[i], y1[order])
        xx2 = np.minimum(x2[i], x2[order])
        yy2 = np.minimum(y2[i], y2[order])

        w, h = np.maximum(xx2 - xx1 + 1, 0), np.maximum(yy2 - yy1 + 1, 0)
        inter = w * h
        iou = inter / (areas[i] + areas[order] - inter)

        inds = np.where(iou < thresh)
        order = order[inds]

    return keep


def image_pyramid(img, min_det_size=12, min_face_size=15,factor= 0.707):
    width, height = img.shape[0], img.shape[1]
    min_length = min(width, height)

    adapt_m = min_det_size / min_face_size
    min_length *= adapt_m

    scale_pyramid = []
    factor_power = 0
    while min_length > min_det_size:
        scale = adapt_m * factor ** factor_power
        scale_pyramid.append(scale)

        min_length *= factor
        factor_power += 1
    return scale_pyramid


def scale_image(img, scale, w, h):
    img_scale = cv2.resize(img, (int(h * scale), int(w * scale)), interpolation=cv2.INTER_LINEAR)
    img_scale = img_scale.astype('float32')
    return img_scale

def save_model(model, model_name, epoch, acc):
    save_path = f'/home/grey/Documents/mtcnn_model_saving/{model_name}/'
    make_sure_path_exists(save_path)
    torch.save(model, os.path.join(save_path, "{}-{}-{:.3f}.pth".format(epoch, model_name, acc)))


def _ohem_mask(loss, ohem_ratio):
    with torch.no_grad():
        values, _ = torch.topk(loss.reshape(-1), int(loss.nelement() * ohem_ratio))
        mask = loss >= values[-1]
    return mask.float()


class CrossEntropyLossWithOHEM(nn.Module):
    def __init__(self,
                 ohem_ratio=1.0,
                 weight=None,
                 ignore_index=-100,
                 eps=1e-7):
        super(CrossEntropyLossWithOHEM, self).__init__()
        self.criterion = nn.CrossEntropyLoss(weight=weight,
                                             ignore_index=ignore_index,
                                             reduction='none')
        self.ohem_ratio = ohem_ratio
        self.eps = eps

    def forward(self, pred, target):
        loss = self.criterion(pred, target)
        mask = _ohem_mask(loss, self.ohem_ratio)
        loss = loss * mask
        return loss.sum() / (mask.sum() + self.eps)

    def set_ohem_ratio(self, ohem_ratio):
        self.ohem_ratio = ohem_ratio


