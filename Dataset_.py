import torch.utils.data as Data
import numpy as np
import os
# import cv2
from PIL import Image
# import torch
import data_processing

IM_SIZE = 224

class Videolist_Parse(object):
    def __init__(self, row):
        self.row = row
    @property
    def path(self):
        return self.row[0]
    @property
    def label(self):
        return int(self.row[1])


class VideoDataset(Data.Dataset):
    def __init__(self, root, list, transform, train_test):

        self.transform = transform
        self.list = list
        self.root = root
        self.train_test = train_test

        self._parse_videolist()


    def __len__(self):
        return len(self.videolist)


    def __getitem__(self, idx):
        record = self.videolist[idx]
        image_tensor = self.get_img(record)

        return image_tensor, record.label-1


    def _parse_videolist(self):
        lines = [x.strip().split(' ') for x in open(self.root + self.list)]
        self.videolist = [Videolist_Parse(item) for item in lines]


    def get_img(self, record):
        dir_img = os.path.join(self.root, self.train_test, record.path.split('.')[0] + '_aligned.jpg')

        feat_traindata = data_processing.feat_data(dir_img)



        image = Image.open(dir_img).convert('RGB')
        # frames = self.transform(image)

        imi = self.transform(image)

        im224 = np.zeros((4, IM_SIZE, IM_SIZE))
        im224[0:3, :, :] = imi
        im224[3, :, :] = feat_traindata

        return im224
