from __future__ import print_function, absolute_import

import os
import torch
import functools
import torch.utils.data as data
from PIL import Image

class VideoDataset(data.Dataset):

    def __init__(self, 
                 dataset, 
                 spatial_transform=None,
                 temporal_transform=None):
        self.dataset = dataset
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (clip, pid, camid) where pid is identity of the clip.
        """
        img_paths, pid, camid, trackid = self.dataset[index]
        # print(len(img_paths))

        if self.temporal_transform is not None:
            img_paths = self.temporal_transform(img_paths)

        clip = []
        for img_path in img_paths:
            img = Image.open(img_path).convert('RGB')  # 3x224x112
            clip.append(img)

   
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        # print(len(clip))
        # trans T x C x H x W to C x T x H x W
        # clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        clip = torch.stack(clip, 0)

        return clip, pid, camid, trackid, img_path.split('/')[-1]
