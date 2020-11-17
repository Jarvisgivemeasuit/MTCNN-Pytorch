import torch
from datasets.wider_face import WIDER
from net.mtcnn import *

class Args:
    def __init__(self):
        self.tr_batch_size = 256
        self.val_batch_size = 256
        self.num_workers = 8

        self.net = 'onet'

        self.trainset = WIDER('train', self.net)
        self.validset = WIDER('val', self.net)

        self.start_epoch = 1
        self.epochs = 120
        self.lr = 0.1
        self.weight_decay = 5e-4

        self.parallel = False
        if self.parallel == True:
            self.gpu_id = '0, 1'
            self.gpu_ids = [0, 1]
        else:
            self.gpu_id = '1'

        self.cuda = torch.cuda.is_available()
        self.apex = False
        self.save_model = True
        self.test_thres = 0.5
