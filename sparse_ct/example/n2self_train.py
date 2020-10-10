
import random

from tqdm import tqdm
import glob
import numpy as np
import torch
from sparse_ct.reconstructor_2d.n2self import Dataset, N2SelfReconstructor


if __name__ == "__main__":
    params= {'batch_size': 5,
            'shuffle': True,
            'num_workers': 5}

    pwd_train = '/external/CT_30_000/train'
    pwd_test = '/external/CT_30_000/test'

    file_list_train = glob.glob(pwd_train+'/*/*/*.png')
    file_list_test = glob.glob(pwd_test+'/*/*/*.png')

    train_loader = torch.utils.data.DataLoader(
        Dataset(
            file_list_train, 
            return_gt=False,
            n_proj=64,
            noise_pow=15.0,
            img_size=512),
        **params
    )

    test_loader = torch.utils.data.DataLoader(
        Dataset(
            random.choices(file_list_test, k=250), 
            return_gt=True,
            n_proj=64,
            noise_pow=15.0,
            img_size=512),
        **params
    )

    theta = np.linspace(0.0, 180.0, 64, endpoint=False)

    recon_n2self = N2SelfReconstructor(
        'N2SelfTrained', theta,
        net='skip', lr=0.001,
        n2self_n_iter=10, 
        n2self_weights=None,
        n2self_proj_ratio=0.2
    )

    for i in range(50):
        print('--------------- ',i)
        recon_n2self._eval(test_loader)
        recon_n2self._train_one_epoch(train_loader, test_loader)
        recon_n2self._save('epoch_{}.pth'.format(i))
    recon_n2self._save('end.pth')

    # for x in tqdm(train_loader):
    #     print(x.shape)
    #     break

    # for x, gt in tqdm(test_loader):
    #     print(x.shape)
    #     print(gt.shape)
    #     break
