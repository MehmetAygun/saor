import os
from functools import lru_cache
from PIL import Image
import numpy as np
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import (ToTensor, Compose, Resize, CenterCrop, functional as Fvision)

from utils.image import square_bbox
from utils.path import DATASETS_PATH
from .torch_transforms import SquarePad, Resize as ResizeCust

import scipy.io as sio
from PIL import ImageFile


ImageFile.LOAD_TRUNCATED_IMAGES = True

PADDING_BBOX = 0.05
BBOX_CROP = True
RANDOM_FLIP = True
SPLIT_DATA = True


class PascalDataset(TorchDataset):

    root = DATASETS_PATH
    name = 'pascal_cow'
    n_channels = 3

    def __init__(self, split, img_size, name):

        self.split = split
        self.name = 'pascal_{}'.format(name)

        if split != 'test':
            assert "Error only for test"

        kp_path = os.path.join(DATASETS_PATH, 'pascal/data/{}_kps.mat'.format(name))

        pascal_anno_path = os.path.join(DATASETS_PATH, 'pascal/data/{}_val.mat'.format(name))
        self.anno = sio.loadmat(pascal_anno_path, struct_as_record=False, squeeze_me=True)['images']

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.net_img_size = (64,64)
        self.bbox_crop = True
        self.resize_mode = 'pad'
        self.padding_mode = 'constant'


    def __len__(self):
        return len(self.anno)


    def __getitem__(self, idx):

        data = self.anno[idx]
        bbox = np.array([data.bbox.x1, data.bbox.y1, data.bbox.x2, data.bbox.y2], float) - 1

        img_path = os.path.join(DATASETS_PATH, 'pascal/VOC2012/JPEGImages', data.rel_path)
        img = Image.open(img_path).convert('RGB')

        kps = data.parts
        v_kps = kps[2,:] == 1.
        not_valid_kps = kps[2,:] != 1.
        kps[:,not_valid_kps] = -1
        kps = kps.T

       
        if self.bbox_crop:
            bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            bbox += np.asarray([round(PADDING_BBOX * s) for s in [-bw, -bh, bw, bh]], dtype=np.int64)
            bbox = square_bbox(bbox.tolist())
            p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
            p_right, p_bottom = max(0, bbox[2] - img.size[0]), max(0, bbox[3] - img.size[1])
            if sum([p_left, p_top, p_right, p_bottom]) > 0:
                img = Fvision.pad(img, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
                bbox = bbox + np.asarray([p_left, p_top, p_left, p_top])
                kps[:,0] += p_left
                kps[:,1] += p_top

            img = img.crop(bbox)
            kps[:,0] = kps[:,0] - np.asarray(bbox[0])
            kps[:,1] = kps[:,1] - np.asarray(bbox[1])
       
        r = self.img_size[0]/img.size[0]*1.
        kps[:,0:2] *= r
        kps[:,0:2] = np.clip(kps[:,0:2],0,self.img_size[0]-1)
        img = self.transform(img)


        net_img = img
        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)

        return {'imgs': img, 'masks': img, 'depths':img, 'poses': poses, 'kps':kps, 'net_imgs':net_img}, -1

    @property
    @lru_cache()
    def transform(self):
        size = self.img_size[0]
        if self.bbox_crop:
            tsfs = [Resize(size), ToTensor()]
        elif self.resize_mode == 'pad':
            tsfs = [ResizeCust(size, fit_inside=True), SquarePad(padding_mode=self.padding_mode), ToTensor()]
        else:
            tsfs = [Resize(size), CenterCrop(size), ToTensor()]
        return Compose(tsfs)

