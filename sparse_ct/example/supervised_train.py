
import random

from tqdm import tqdm
import glob
import numpy as np
import torch
from sparse_ct.reconstructor_2d.dataset import (
                DeepLesionDataset, EllipsesDataset )
from sparse_ct.reconstructor_2d.supervised import SupervisedReconstructor            


if __name__ == "__main__":

    params= {'batch_size': 8,
            'shuffle': True,
            'num_workers': 5}

    # pwd_train = '/external/CT_30_000/train'
    # pwd_test = '/external/CT_30_000/test'

    # file_list_train = glob.glob(pwd_train+'/*/*/*.png')
    # file_list_test = glob.glob(pwd_test+'/*/*/*.png')

    # train_loader = torch.utils.data.DataLoader(
    #     DeepLesionDataset(
    #         file_list_train, 
    #         return_gt=True,
    #         n_proj=64,
    #         img_size=512),
    #     **params
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     DeepLesionDataset(
    #         random.choices(file_list_test, k=250), 
    #         return_gt=True,
    #         n_proj=64,
    #         img_size=512),
    #     **params
    # )

    train_loader = torch.utils.data.DataLoader(
        EllipsesDataset(
            ellipses_type='train', 
            return_gt=True,
            n_proj=64,
            img_size=512),
        **params
    )

    test_loader = torch.utils.data.DataLoader(
        EllipsesDataset(
            ellipses_type='validation',
            return_gt=True,
            n_proj=64,
            img_size=512),
        **params
    )

    theta = np.linspace(0.0, 180.0, 64, endpoint=False)

    recon_supervised = SupervisedReconstructor(
        'Supervised',
        net='unet', lr=0.0001, weights=None
    )
    recon_supervised.init_train(theta)

    for i in range(50):
        print('--------------- ',i)
        recon_supervised._train_one_epoch(train_loader, test_loader)
        recon_supervised._eval(test_loader)
        recon_supervised._save('epoch_{}.pth'.format(i))
    recon_supervised._save('end.pth')

    # for x in tqdm(train_loader):
    #     print(x.shape)
    #     break

    # for x, gt in tqdm(test_loader):
    #     print(x.shape)
    #     print(gt.shape)
    #     break
