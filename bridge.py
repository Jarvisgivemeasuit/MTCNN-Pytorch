from math import factorial
import numpy as np
import os
from PIL import Image
from progress.bar import Bar

import torch
import torch.nn.functional as F
import albumentations as A

from utils.utils import *
from net.mtcnn import *


class Bridge:
    def __init__(self, mode='train', net_dataset='onet'):
        self.mode = mode
        self.net = net_dataset

        self.pnet = torch.load('/home/grey/Documents/mtcnn_model_saving/pnet/91-pnet-0.911.pth')
        self.rnet = torch.load('/home/grey/Documents/mtcnn_model_saving/rnet/111-rnet-0.910.pth')
        # self.onet = torch.load('/home/grey/Documents/mtcnn_model_saving/onet/104-onet-0.910.pth')

        # self.pnet = PNet().cuda()
        # weights = torch.load('/home/grey/Documents/mtcnn_model_saving/pnet_timesler.pt')
        # self.pnet.load_state_dict(weights)

        self.img_path = f'/data/grey/WIDER_FACE/WIDER_{mode}/images'
        self.label_path = f'/data/grey/WIDER_FACE/WIDER_{mode}/labels'
        self.save_path = f'/data/grey/WIDER_FACE/{self.net}_{mode}set/'

        self.mean = [0.45650857, 0.39260386, 0.36109988]
        self.std = [0.27675595, 0.26248012, 0.26266556]
    
        self.norm = A.Normalize(mean=self.mean, std=self.std, p=1)

        self.faces_count = 1
        self.pos_count = 1
        self.par_count = 1
        self.neg_count = 1

        self.labels = []

    def generate_dataset(self, face_thres=[0.6, 0.7, 0.8], nms_thres=[0.7, 0.7, 0.7]):
        cate_list = os.listdir(self.img_path)
        bar = Bar(f'generating {self.net}\'s {self.mode}set:', max=99999)

        for cate in cate_list:
            img_list = os.listdir(os.path.join(self.img_path, cate))

            for img_file in img_list:
                label_file = img_file.replace('jpg', 'npy')
                img = np.array(Image.open(os.path.join(self.img_path, cate, img_file)))
                gts = np.load(os.path.join(self.label_path, cate, label_file))[:4]

                bboxes = self.pnet_output(img, face_thres[0], nms_thres[0])
                if self.net == 'onet':
                    bboxes = self.rnet_output(bboxes, img, face_thres[1], nms_thres[1])

                self.save_net_set(bboxes, img, gts, bar, self.net)

        np.save(os.path.join(self.save_path, 'label_list_net'), self.labels)
        bar.finish()

    def pnet_output(self, img, face_threshold, nms_threshold):
        img_w, img_h = img.shape[0], img.shape[1]
        scale_pyramid = image_pyramid(img)

        bboxes = []

        for scale in scale_pyramid:
            img_scale = scale_image(img, scale, img_w, img_h)
            img_ = self._img2normalize_tensor(img_scale)

            with torch.no_grad():
                [pred_cls, bbox_reg] = self.pnet(img_)
                # [bbox_reg, pred_cls] = self.pnet(img_)

            dets = self._scale_img_boxes(pred_cls, bbox_reg, scale, face_threshold)
            bboxes.append(dets)

        bboxes = np.vstack(bboxes)
        keep = nms(bboxes, nms_threshold)

        return bboxes[keep]

    def rnet_output(self, bboxes, img, face_threshold, nms_threshold):
        img_w, img_h = img.shape[0], img.shape[1]

        faces = torch.zeros((bboxes.shape[0], 3, 24, 24)).cuda()
        for idx, bbox in enumerate(bboxes):
            face, _ = self.generate_face(img, bbox, img_w, img_h, (24, 24))
            faces[idx] = self._img2normalize_tensor(face)
        
        with torch.no_grad():
            [pred_cls, bbox_reg] = self.rnet(faces)

        pred_cls = F.softmax(pred_cls.cpu(), dim=1)
        bboxes = bboxes[np.where(pred_cls[:, 1] > face_threshold)]

        bbox_reg = np.array(bbox_reg.cpu())[np.where(pred_cls[:, 1] > face_threshold)]
        bboxes = self.fine_tuning(bboxes, bbox_reg)

        keep = nms(bboxes, nms_threshold)

        return bboxes[keep]

    def onet_output(self, bboxes, img, face_threshold, nms_threshold):
        img_w, img_h = img.shape[0], img.shape[1]

        faces = torch.zeros((bboxes.shape[0], 3, 48, 48)).cuda()
        for idx, bbox in enumerate(bboxes):
            face, _ = self.generate_face(img, bbox, img_w, img_h, (48, 48))
            faces[idx] = self._img2normalize_tensor(face)
        
        with torch.no_grad():
            [pred_cls, bbox_reg] = self.onet(faces)

        pred_cls = F.softmax(pred_cls.cpu(), dim=1)
        bboxes = bboxes[np.where(pred_cls[:, 1] > face_threshold)]

        bbox_reg = np.array(bbox_reg.cpu())[np.where(pred_cls[:, 1] > face_threshold)]
        bboxes = self.fine_tuning(bboxes, bbox_reg)

        keep = nms(bboxes, nms_threshold)
        return bboxes[keep]

    def _img2normalize_tensor(self, img):
        img = self.norm(image=img)['image']
        img = np.expand_dims(img.transpose(2, 0, 1), 0)
        img_ = torch.tensor(img).cuda()
        return img_

    def _scale_img_boxes(self, pred_cls, bbox_reg, scale, thres):
        pred_cls = pred_cls.squeeze(0).cpu()
        pred_cls = F.softmax(pred_cls, dim=0)

        bbox_reg = np.squeeze(bbox_reg.cpu().numpy().transpose(2, 3, 0, 1), axis=2)
        bbox_reg = bbox_reg[np.where(pred_cls[1] > thres)]

        y, x = np.array(np.where(pred_cls[1] > thres)) * 2
        w, h = np.ones(y.shape) * 12, np.ones(x.shape) * 12
        score = pred_cls[1][pred_cls[1] > thres].flatten()

        dets = np.vstack([x, y, w, h, score]).transpose()
        keep = nms(dets, 0.5)

        dets = dets[keep]
        bbox_reg = bbox_reg[keep]

        dets = self.fine_tuning(dets, bbox_reg)
        dets[:, :4] = dets[:, :4] / scale

        return dets

    def fine_tuning(self, dets, bbox_reg):
        dets[:, 0] = dets[:, 0] + bbox_reg[:, 0] * dets[:, 2]
        dets[:, 1] = dets[:, 1] + bbox_reg[:, 1] * dets[:, 3]
        dets[:, 2] = dets[:, 2] * np.exp(bbox_reg[:, 2])
        dets[:, 3] = dets[:, 3] * np.exp(bbox_reg[:, 3])
        return dets

    def save_net_set(self, bboxes, img, gts, bar, net):
        img_w, img_h = img.shape[0], img.shape[1]
        size = (24, 24) if net == 'rnet' else (48, 48)
        img_size = '24x24' if net == 'rnet' else '48x48'

        for bbox in bboxes:
            face, label = self.generate_img_and_label(img, gts, bbox, img_w, img_h, size)

            if self.neg_count > self.pos_count * 2 and label[-1] == 0:
                continue
            if self.par_count > self.pos_count and label[-1] == -1:
                continue

            make_sure_path_exists(os.path.join(self.save_path, f'img{img_size}'))
            np.save(os.path.join(self.save_path, f'img{img_size}', f'{self.faces_count}'), face)

            self.labels.append(label)

            if label[-1] == 0:
                self.neg_count += 1
            elif label[-1] == 1:
                self.pos_count += 1
            elif label[-1] == -1:
                self.par_count += 1

            bar.suffix = f'{self.faces_count} / unknown'
            bar.next()

            self.faces_count += 1

    def generate_img_and_label(self, img, gts, bbox, img_w, img_h, size):
        face, bbox = self.generate_face(img, bbox, img_w, img_h, size)

        bbox_ = bbox.reshape(-1, 5)
        ious = iou(bbox_, gts)

        # negitive sample
        label = np.zeros(5)

        # positive sample
        idx = np.where(ious > 0.6)[0]
        if len(idx):
            gt = gts[idx][0]
            label[-1] = 1

            label[0] = (gt[0] - bbox[0]) / bbox[2]
            label[1] = (gt[1] - bbox[1]) / bbox[3]
            label[2] = np.log(gt[2] / bbox[2])
            label[3] = np.log(gt[3] / bbox[3])

        # part sample
        elif len(np.where(ious > 0.3)[0]):
            idx = np.where(ious > 0.3)[0]

            gt = gts[idx][0]
            label[-1] = -1

            label[0] = (gt[0] - bbox[0]) / bbox[2]
            label[1] = (gt[1] - bbox[1]) / bbox[3]
            label[2] = np.log(gt[2] / bbox[2])
            label[3] = np.log(gt[3] / bbox[3])

        return face, label

    def generate_face(self, img, bbox, img_w, img_h, size):
        bbox = bbox.astype('int')

        side = max(bbox[2], bbox[3])
        bbox[2], bbox[3] = side, side

        face = img[max(bbox[1], 0):min(bbox[1] + bbox[2], img_w), max(bbox[0], 0):min(bbox[0] + bbox[3], img_h)]

        if 0 > bbox[1]:
            face = np.pad(face, ((-bbox[1], 0), (0, 0), (0, 0)), 'constant')
        if 0 > bbox[0]:
            face = np.pad(face, ((0, 0), (-bbox[0], 0), (0, 0)), 'constant')
        if bbox[1] + bbox[2] > img_w:
            face = np.pad(face, ((0, bbox[1] + bbox[2] - img_w), (0, 0), (0, 0)), 'constant')
        if bbox[0] + bbox[3] > img_h:
            face = np.pad(face, ((0, 0), (0, bbox[0] + bbox[3] - img_h), (0, 0)), 'constant')

        face = Image.fromarray(face)
        face = np.array(face.resize(size))
        return face, bbox

def resize_img(img_path, save_path, size=(12, 12)):
    file_list = os.listdir(img_path)
    num_imgs = len(file_list)
    bar = Bar('resizing images:', max=num_imgs)

    for idx, file in enumerate(file_list):
        img = np.array(Image.open(os.path.join(img_path, file)))
        img = cv2.resize(img, size, interpolation=cv2.INTER_LINEAR)
        img = Image.fromarray(img)
        img.save(os.path.join(save_path, file))
        bar.suffix = f'{idx + 1} / {num_imgs}'
        bar.next()
    bar.finish()


def detector(img, face_thres=[0.6, 0.7, 0.8], nms_thres=[0.7, 0.7, 0.2]):
    bridge = Bridge()
    bboxes = bridge.pnet_output(img, face_thres[0], nms_thres[0])
    bboxes = bridge.rnet_output(bboxes, img, face_thres[1], nms_thres[1])
    bboxes = bridge.onet_output(bboxes, img, face_thres[2], nms_thres[2])
    return bboxes


if __name__ == '__main__':
    model = 'onet'
    dataset_type = 'val'

    bridge = Bridge(mode=dataset_type, net_dataset=model)
    bridge.generate_dataset()

# img_path = f'/data/grey/WIDER_FACE/{model}_{dataset_type}set/img'
# save_path = f'/data/grey/WIDER_FACE/{model}_{dataset_type}set/img24x24'
# resize_img(img_path, save_path, (24, 24))