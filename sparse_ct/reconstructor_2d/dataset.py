

from sparse_ct.data import (image_to_sparse_sinogram, 
                        ellipses_to_sparse_sinogram)
import numpy as np
import torch
import random

class DeepLesionDataset(torch.utils.data.Dataset):
    def __init__(self, file_list, 
                return_gt=False,
                n_proj=32,
                noise_pow=np.linspace(30.0, 40.0, 100),
                img_size=512):
        self.file_list = file_list
        self.return_gt = return_gt
        self.noise_pow = noise_pow
        self.n_proj = n_proj
        self.size = img_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # get image
        gt, sinogram, _, _ = image_to_sparse_sinogram(
            self.file_list[index],
            gray=True,
            n_proj=self.n_proj,
            channel=1,
            size=self.size,
            noise_pow=random.choice(
                self.noise_pow
                )
            )
        # get projs
        if self.return_gt:
            return np.expand_dims(sinogram, axis=0), np.expand_dims(gt, axis=0)
        else:
            return np.expand_dims(sinogram, axis=0)      

class EllipsesDataset(torch.utils.data.Dataset):
    def __init__(self, ellipses_type, 
                return_gt=False,
                n_proj=32,
                noise_pow=np.linspace(30.0, 40.0, 100),
                img_size=512):
        self.ellipses_type = ellipses_type
        self.return_gt = return_gt
        self.noise_pow = noise_pow
        self.n_proj = n_proj
        self.size = img_size

    def __len__(self):
        if self.ellipses_type == 'train':
            return 32000
        elif self.ellipses_type == 'validation':
            return 1000

    def __getitem__(self, index):
        # get image
        gt, sinogram, _, _ = ellipses_to_sparse_sinogram(
            part=self.ellipses_type,
            gray=True,
            n_proj=self.n_proj,
            channel=1,
            size=self.size,
            noise_pow=random.choice(
                self.noise_pow
                )
            )
        # get projs
        if self.return_gt:
            return np.expand_dims(sinogram, axis=0), np.expand_dims(gt, axis=0)
        else:
            return np.expand_dims(sinogram, axis=0)