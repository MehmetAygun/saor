import os,sys
from copy import deepcopy
from functools import lru_cache
from PIL import Image

import numpy as np
import pandas as pd
import scipy.io as sio
import random
import torch
from torch.utils.data.dataset import Dataset as TorchDataset
from torchvision.transforms import (ToTensor, Compose, Resize, RandomCrop, CenterCrop, functional as Fvision,
                                    RandomHorizontalFlip)

from utils import path_exists, get_files_from, use_seed
from utils.image import square_bbox
from utils.path import DATASETS_PATH
from .torch_transforms import SquarePad, Resize as ResizeCust

PADDING_BBOX = 0.05
JITTER_BBOX = 0.05
BBOX_CROP = True
RANDOM_FLIP = False
RANDOM_JITTER = False
SPLIT_DATA = True


class CUB200Dataset(TorchDataset):
    root = DATASETS_PATH
    name = 'cub_200'
    n_channels = 3

    def __init__(self, split, img_size, **kwargs):

        kwargs = deepcopy(kwargs)
        self.split = split
        self.data_path = path_exists(DATASETS_PATH / 'cub_200' / 'images')
        self.input_files = get_files_from(self.data_path, ['png', 'jpg'], recursive=True, sort=True)

        self.weighted_sample = True

        anno_path = os.path.join(DATASETS_PATH, 'cub_200', '{}_cub_cleaned.mat'.format(split))
        anno = sio.loadmat(
            anno_path, struct_as_record=False, squeeze_me=True)['images']
        self.masks = {}
        for data in anno:
            self.masks[data.rel_path] = data.mask

        root = self.data_path.parent
        filenames = pd.read_csv(root / 'images.txt', sep=' ', index_col=0, header=None)[1].tolist()

        if self.weighted_sample == True and self.split == 'train':
            names, clusters = np.load(os.path.join(DATASETS_PATH, 'cub_200', 'cub_clusters_10.npy'))
            cs, counts  = np.unique(clusters, return_counts=True)
            self.clusters_to_names = dict.fromkeys(cs, [])
            for n,c in zip(names, clusters):
                self.clusters_to_names[c].append(n)
            counts = 1/counts
            counts = counts / np.sum(counts)
            self.clusters = cs
            self.cluster_freqs = counts

        with open(os.path.join(DATASETS_PATH, 'cub_200', 'parts', 'part_locs.txt')) as file:
            parts = file.readlines()
            parts = [part.rstrip() for part in parts]
            self.kps = {part.split(' ')[0]: [] for part in parts}
            for part in parts:
                p = part.split(' ')
                kp = [float(c) for c in p[2:]]
                self.kps[p[0]].append(kp)

        bboxes = pd.read_csv(root / 'bounding_boxes.txt', sep=' ', index_col=0, header=None).astype(int)
        bboxes[3], bboxes[4] = bboxes[1] + bboxes[3], bboxes[2] + bboxes[4]  # XXX bbox format before is [x, y, w, h]
        self.bbox_mapping = {filenames[k]: bboxes.iloc[k].tolist() for k in range(len(filenames))}
        self.kps_mapping = {filenames[k]: self.kps[str(k+1)] for k in range(len(filenames))}

        assert len(self.bbox_mapping) == len(self.input_files)
        self.input_files = [x for x in self.input_files if str(x.relative_to(self.data_path)) in self.masks]

        if self.split in ['val', 'test']:  # XXX images are sorted by model so we shuffle
            with use_seed(123):
                np.random.shuffle(self.input_files)

        if self.split =='train':
            self.path_to_idx = {}
            for idx, path in enumerate(self.input_files):
                self.path_to_idx[str(self.input_files[idx].relative_to(self.data_path))] = idx

        self.img_size = (img_size, img_size) if isinstance(img_size, int) else img_size
        self.bbox_crop = kwargs.pop('bbox_crop', True)
        self.resize_mode = kwargs.pop('resize_mode', 'pad')
        assert self.resize_mode in ['crop', 'pad']
        self.padding_mode = kwargs.pop('padding_mode', 'edge')
        self.random_flip = kwargs.pop('random_flip', RANDOM_FLIP)
        self.random_jitter = kwargs.pop('random_jitter', RANDOM_JITTER)
        self.random_crop = kwargs.pop('random_crop', False) and split == 'train'
        assert len(kwargs) == 0, kwargs

    def __len__(self):
        return len(self.input_files) if self.split != 'val' else 5

    def __getitem__(self, idx):

        if self.weighted_sample and self.split == 'train':
            selected_cluster = np.random.choice(self.clusters, 1, p=self.cluster_freqs)[0]
            rel_path = np.random.choice(self.clusters_to_names[selected_cluster],1)[0]
            img_path = os.path.join(self.data_path, rel_path)

        else:
            rel_path = str(self.input_files[idx].relative_to(self.data_path))
            img_path = str(self.input_files[idx])

        img = Image.open(img_path).convert('RGB')
        depth = Image.open(img_path.replace('images','depths').replace('.jpg', '.png'))

        kps = np.asarray(self.kps_mapping[rel_path])
        mask = self.masks[rel_path]
        mask = Image.fromarray(mask)

        if self.bbox_crop:
            bbox = np.asarray(self.bbox_mapping[rel_path])
            bw, bh = bbox[2] - bbox[0] + 1, bbox[3] - bbox[1] + 1
            bbox += np.asarray([round(PADDING_BBOX * s) for s in [-bw, -bh, bw, bh]], dtype=np.int64)
            if self.random_jitter and self.split == 'train':
                bbox += np.asarray([round(JITTER_BBOX * s * (1-2*random.random())) for s in [bw, bh, bw, bh]], dtype=np.int64)
            bbox = square_bbox(bbox.tolist())
            p_left, p_top = max(0, -bbox[0]), max(0, -bbox[1])
            p_right, p_bottom = max(0, bbox[2] - img.size[0]), max(0, bbox[3] - img.size[1])
            if sum([p_left, p_top, p_right, p_bottom]) > 0:
                img = Fvision.pad(img, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
                depth = Fvision.pad(depth, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
                mask = Fvision.pad(mask, (p_left, p_top, p_right, p_bottom), padding_mode=self.padding_mode)
            adj_bbox = bbox + np.asarray([p_left, p_top, p_left, p_top])

            img = img.crop(adj_bbox)
            mask = mask.crop(adj_bbox)
            depth = depth.crop(adj_bbox)

            visible = kps[:, 2].astype(bool)
            kps[visible, :2] -= np.asarray([bbox[0], bbox[1]], dtype=np.float32)

        r = self.img_size[0]/img.size[0]*1.
        kps[:,0:2] *= r
        kps[:,0:2] = np.clip(kps[:,0:2],0,img.size[0]-1)

        img = self.transform(img)
        mask = self.transform(mask)
        mask = (mask> 0)*1.

        depth = self.transform(depth)
       
        net_img = img
        poses = torch.cat([torch.eye(3), torch.Tensor([[0], [0], [2.732]])], dim=1)

        return {'imgs': img, 'masks': mask, 'depths':depth, 'poses': poses, 'kps':kps, 'net_imgs':net_img}, -1
       
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
