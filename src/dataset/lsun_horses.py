import os
from copy import deepcopy
from functools import lru_cache
from PIL import Image
import numpy as np
from random import random, shuffle, seed

import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import (ToTensor, Compose, Resize, RandomCrop, CenterCrop, functional as Fvision,
                                    RandomHorizontalFlip)

from utils.image import square_bbox
from utils.path import DATASETS_PATH
from .torch_transforms import SquarePad, Resize as ResizeCust


seed(42)

PADDING_BBOX = 0.05
JITTER_BBOX = 0.05
BBOX_CROP = True
RANDOM_FLIP = True
RANDOM_JITTER = True
SPLIT_DATA = True

class LsunHorsesDataset(TorchDataset):

    root = DATASETS_PATH
    name = 'lsun_horses'
    n_channels = 3

    def __init__(self, split, img_size, **kwargs):
        kwargs = deepcopy(kwargs)
        self.weighted_sample = True

        self.split = split
        with open(os.path.join(DATASETS_PATH, 'lsun_horses', 'train_clean.txt')) as f:
            data = f.readlines()
            data = [d.strip() for d in data]

        if self.weighted_sample == True and self.split == 'train':
            clusters = np.load(os.path.join(DATASETS_PATH, 'lsun_horses', 'clusters_10_new_clean.npy'))
            assert len(data) == clusters.shape[0]
            cs, counts  = np.unique(clusters, return_counts=True)

            counts = 1/counts
            counts = counts / np.sum(counts)
            self.clusters = cs

            self.cluster_freqs = counts
            data = [d + ' ' + str(c) for d,c in zip(data, clusters)]

        shuffle(data)
        n_val = int(len(data)* 0.95)

        if self.split in ['val', 'test']:  # XXX images are sorted by model so we shuffle
            self.data = data[n_val:]
        else:
            self.data = data[:n_val]

        if self.weighted_sample == True and self.split == 'train':
            self.clusters_to_idxs = {}
            for c in self.clusters:
                self.clusters_to_idxs[c] = []

            for idx, d in enumerate(self.data):
                c = self.data[idx].split(' ')[-1]
                self.clusters_to_idxs[int(c)].append(idx)

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.net_img_size = (64,64)
        self.bbox_crop = kwargs.pop('bbox_crop', True)
        self.resize_mode = kwargs.pop('resize_mode', 'pad')
        assert self.resize_mode in ['crop', 'pad']
        self.padding_mode = kwargs.pop('padding_mode', 'constant')

        self.random_flip = kwargs.pop('random_flip', False)
        self.random_jitter = kwargs.pop('random_jitter', RANDOM_JITTER)
        self.random_crop = kwargs.pop('random_crop', False) and split == 'train'
        assert len(kwargs) == 0, kwargs

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):

        if self.weighted_sample and self.split == 'train':
            selected_cluster = np.random.choice(self.clusters, 1)[0]
            idx = np.random.choice(self.clusters_to_idxs[selected_cluster],1)[0]
            path, x1,y1,x2,y2,c = self.data[idx].split(' ')
        else:
            path, x1,y1,x2,y2 = self.data[idx].split(' ')
            selected_cluster = 0

        x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)

        img_path = os.path.join(DATASETS_PATH, 'lsun_horses/images', path)
        mask_path = os.path.join(DATASETS_PATH, 'lsun_horses/masks', path.replace('.jpg','_{}_{}.npy'.format(x1,y1)))
        depth_path = os.path.join(DATASETS_PATH, 'lsun_horses/depths', path.replace('.jpg','-dpt_beit_large_512.png'))

        img = Image.open(img_path).convert('RGB')
        mask = np.load(mask_path).astype(np.uint8)

        mask = Image.fromarray(mask)
        depth = Image.open(depth_path)

        if self.bbox_crop:
            bbox = np.asarray([x1,y1,x2,y2])
            bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            bbox += np.asarray([round(PADDING_BBOX * s) for s in [-bw, -bh, bw, bh]], dtype=np.int64)
            if self.random_jitter and self.split == 'train':
                jitter = np.asarray([round(JITTER_BBOX * s * (1-2*random())) for s in [bw, bh, bw, bh]], dtype=np.int64)
                bbox += jitter
            bbox = square_bbox(bbox.tolist())
            p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
            p_right, p_bottom = max(0, bbox[2] - img.size[0]), max(0, bbox[3] - img.size[1])
            if sum([p_left, p_top, p_right, p_bottom]) > 0:
                img = Fvision.pad(img, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
                mask = Fvision.pad(mask, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
                depth = Fvision.pad(depth, (p_left, p_top, p_right, p_bottom), padding_mode='edge')
                bbox = bbox + np.asarray([p_left, p_top, p_left, p_top])

            img = img.crop(bbox)
            mask = mask.crop(bbox)
            depth = depth.crop(bbox)
           
        img = self.transform(img)
        mask = self.transform(mask)
        depth = self.transform(depth)
      
        mask = (mask> 0)*1.
        net_img = img
        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)
        return {'imgs': img, 'masks': mask, 'depths':depth, 'depths_c':-1, 'poses': poses, 'kps':-1, 'net_imgs':net_img, 'dino_pca':-1, 'cluster': torch.tensor([selected_cluster])}, -1

    @property
    @lru_cache()
    def transform(self):
        size = self.img_size[0]
        if self.bbox_crop:
            tsfs = [Resize(size), ToTensor()]
        elif self.resize_mode == 'pad':
            tsfs = [ResizeCust(size, fit_inside=True), SquarePad(padding_mode=self.padding_mode), ToTensor()]
        elif self.random_crop:
            tsfs = [Resize(size), RandomCrop(size), ToTensor()]
        else:
            tsfs = [Resize(size), CenterCrop(size), ToTensor()]
        if self.random_flip and self.split == 'train':
            tsfs = [RandomHorizontalFlip()] + tsfs
        return Compose(tsfs)

