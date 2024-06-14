import os
from copy import deepcopy
from functools import lru_cache
from PIL import Image
import numpy as np
from random import random, shuffle
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import (ToTensor, Compose, Resize, RandomCrop, CenterCrop, functional as Fvision,
                                    RandomHorizontalFlip)

from utils.image import square_bbox
from utils.path import DATASETS_PATH
from .torch_transforms import SquarePad, Resize as ResizeCust


PADDING_BBOX = 0.05
JITTER_BBOX = 0.05
BBOX_CROP = True
RANDOM_FLIP = True
RANDOM_JITTER = True
SPLIT_DATA = True


class INatMammalNetlDataset(TorchDataset):

    root = DATASETS_PATH
    name = 'inat_mix'
    n_channels = 3

    def __init__(self, split, img_size, **kwargs):
        kwargs = deepcopy(kwargs)
        self.split = split
        self.name = 'inat_mix'
        self.weighted_sample = True

        # load all data
        with open(os.path.join(DATASETS_PATH, self.name, 'train_inat.txt')) as f:   
            data = f.readlines()
            data = [d.strip() for d in data]

        # get taxa data
        self.tax_to_data = {}
        for d in data:
            img_path = d.split(' ')[0]
            tax = img_path.split('/')[-2]
            if tax not in self.tax_to_data:
                self.tax_to_data[tax] = []
            self.tax_to_data[tax].append(d)

        self.taxs = list(self.tax_to_data.keys())
        # select taxs via number of images
        self.taxs = [tax for tax in self.taxs if len(self.tax_to_data[tax]) > 130]
        
        print (self.taxs, len(self.taxs))
        self.n_classes = len(self.taxs)
        if self.weighted_sample == True and self.split == 'train':
            self.clusters_to_idxs = {}
            for tax in self.taxs:
                self.clusters_to_idxs[tax] = {}
                clusters = np.load(os.path.join(DATASETS_PATH, self.name, 'clusters', tax, 'clusters_10.npy'))

                assert (len(self.tax_to_data[tax]) == clusters.shape[0])
                for cs in np.unique(clusters):
                    self.clusters_to_idxs[tax][cs] = []
                self.tax_to_data[tax]= [data + ' ' + str(cluster)  for data, cluster in zip(self.tax_to_data[tax], clusters)]

        self.n_samples = 0
        for tax in self.taxs:
            shuffle(self.tax_to_data[tax])
            n_val = int(len(self.tax_to_data[tax])* 0.99)
            if self.split in ['val', 'test']:  # XXX images are sorted by model so we shuffle
                self.tax_to_data[tax] = self.tax_to_data[tax][n_val:]
                self.n_samples += len(self.tax_to_data[tax])
            else:
                self.tax_to_data[tax] = self.tax_to_data[tax][:n_val]
                self.n_samples += len(self.tax_to_data[tax])

        if self.weighted_sample == True and self.split == 'train':
            for tax in self.taxs:
                for idx, data in enumerate(self.tax_to_data[tax]):
                    img_path, x1,y1,x2,y2, c = data.split(' ')
                    self.clusters_to_idxs[tax][int(c)].append(idx)

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.net_img_size = (64,64)
        self.bbox_crop = kwargs.pop('bbox_crop', True)
        self.resize_mode = kwargs.pop('resize_mode', 'pad')
        assert self.resize_mode in ['crop', 'pad']
        self.padding_mode = kwargs.pop('padding_mode', 'edge')
        self.random_flip = kwargs.pop('random_flip', False)
        self.random_jitter = kwargs.pop('random_jitter', RANDOM_JITTER)
        self.random_crop = kwargs.pop('random_crop', False) and split == 'train'
        assert len(kwargs) == 0, kwargs


    def __len__(self):
        return self.n_samples if self.split == 'train' else 8

    def __getitem__(self, idx):

        c_id = np.random.randint(len(self.taxs))
        cat = self.taxs[c_id]

        if self.weighted_sample == True and self.split == 'train':
            cluster = np.random.choice(list(self.clusters_to_idxs[cat].keys()), 1)[0]
            idx = np.random.choice(self.clusters_to_idxs[cat][cluster],1)[0]
            data = self.tax_to_data[cat][idx]
        else:
            data = np.random.choice(self.tax_to_data[cat], 1)[0]

        if self.weighted_sample == True and self.split == 'train':
            path, x1,y1,x2,y2,c = data.split(' ')
        else:
            path, x1,y1,x2,y2 = data.split(' ')

        name = path.split('/')[-1]
        img_path = os.path.join(DATASETS_PATH, self.name, 'images', cat, name)
        img = Image.open(img_path).convert('RGB')

        mask_path = os.path.join(DATASETS_PATH, self.name, 'masks_png', cat, name.replace('.jpg','_{}_{}.png'.format(x1,y1)))
        mask = Image.open(mask_path).convert('L')

        depth_name = os.path.join(DATASETS_PATH, self.name, 'depths', cat, name.replace('.jpg', '-dpt_beit_large_512.png'))
        depth = Image.open(depth_name)

        x1,y1,x2,y2 = float(x1), float(y1), float(x2), float(y2)

        if self.bbox_crop:
            bbox = np.asarray([x1,y1,x2,y2])
            bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            bbox += np.asarray([round(PADDING_BBOX * s) for s in [-bw, -bh, bw, bh]], dtype=np.int64)
            if self.random_jitter and self.split == 'train':
                bbox += np.asarray([round(JITTER_BBOX * s * (1-2*random())) for s in [bw, bh, bw, bh]], dtype=np.int64)
            bbox = square_bbox(bbox.tolist())
            p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
            p_right, p_bottom = max(0, bbox[2] - img.size[0]), max(0, bbox[3] - img.size[1])
            if sum([p_left, p_top, p_right, p_bottom]) > 0:
                img = Fvision.pad(img, (p_left, p_top, p_right, p_bottom), padding_mode='constant')
                mask = Fvision.pad(mask, (p_left, p_top, p_right, p_bottom), padding_mode='constant')
                depth = Fvision.pad(depth, (p_left, p_top, p_right, p_bottom), padding_mode='edge')
                bbox = bbox + np.asarray([p_left, p_top, p_left, p_top])

            img = img.crop(bbox)
            mask = mask.crop(bbox)
            depth = depth.crop(bbox)

        img = self.transform(img)
        mask = self.transform(mask)
        depth = self.transform(depth)

        mask = (mask> 0)*1.

        if self.split == 'train' and random()>0.5:
            img = Fvision.hflip(img)
            depth = Fvision.hflip(depth)
            mask = Fvision.hflip(mask)

        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)

        return {'imgs': img, 'masks': mask, 'depths':depth, 'poses': poses, 'cats':torch.tensor(c_id)}, -1


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
