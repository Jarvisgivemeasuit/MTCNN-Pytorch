import os
import time
from albumentations.augmentations.transforms import Resize
import numpy as np
import albumentations as A
from PIL import Image
from datasets.wider_face_utils import *
from torch.utils.data import Dataset


class WIDER(Dataset):
    def __init__(self, mode='train', net='pnet', base_dir = '/data/grey/WIDER_FACE/'):
        super().__init__()
        self.mode = mode
        self.net = net
        if net == 'pnet':
            img_size = '12x12'
        elif net == 'rnet':
            img_size = '24x24'
        else:
            img_size = '48x48'

        self.mean = [0.45781077, 0.37942231, 0.34568618]
        self.std = [0.26442264, 0.24752007, 0.2465438]

        self.img_dir = os.path.join(base_dir, f'{net}_{mode}set/img{img_size}')
        # self.label_list = np.load(os.path.join(base_dir, f'{net}_{mode}set/label_list.npy'))
        self.label_list = np.load(os.path.join(base_dir, f'{net}_{mode}set/label_list_data.npy'))

    def __len__(self):
        return len(self.label_list)

    def __getitem__(self, idx):
        return self.get_data(idx, self.mode, self.net)

    def get_data(self, idx, mode, net):
        # img = np.array(Image.open(os.path.join(self.img_dir, f'{idx+1}.jpg')))
        # start = time.time()
        # img = np.load(os.path.join(self.img_dir, f'{idx+1}.npy'))
        if mode == 'train':
            img = np.load(os.path.join(self.img_dir, f'{idx + 120941}.npy'))
            img = self.tr_normalization(img).transpose(2, 0, 1)
        else:
            img = np.load(os.path.join(self.img_dir, f'{idx + 30360}.npy'))
            img = self.vd_normalization(img).transpose(2, 0, 1)

        offset_label = self.label_list[idx][:4].astype('float32')
        cls_label = self.label_list[idx][4].astype('int')

        if net == 'pnet':
            offset_label = offset_label.reshape(-1, 1, 1)
            cls_label = cls_label.reshape(-1, 1)
        # print(time.time() - start)
        return img, offset_label, cls_label 

    def tr_normalization(self, img):
        norm = A.Compose([
            A.RGBShift(p=0.6),
            A.ChannelShuffle(p=0.6),
            A.RandomBrightnessContrast(p=0.6),
            A.Normalize(mean=self.mean, std=self.std, p=1)]
        )
        return norm(image=img)['image']

    def vd_normalization(self, img):
        norm = A.Compose([
            A.Normalize(mean=self.mean, std=self.std, p=1)]
        )
        return norm(image=img)['image']