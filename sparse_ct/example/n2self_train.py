
import random

from tqdm import tqdm
import glob
import numpy as np
import torch
from sparse_ct.reconstructor_2d.n2self import (
                N2SelfReconstructor)
from sparse_ct.reconstructor_2d.dataset import (
                DeepLesionDataset, EllipsesDataset)


if __name__ == "__main__":

    params= {'batch_size': 8,
            'shuffle': True,
            'num_workers': 8}
    N_PROJ = 64
    pwd_train = '/external/CT_30_000/train'
    pwd_test = '/external/CT_30_000/test'

    file_list_train = glob.glob(pwd_train+'/*/*/*/*.png')
    file_list_test = glob.glob(pwd_test+'/*/*/*/*.png')
    print("file_list_train", len(file_list_train))
    print("file_list_test", len(file_list_test))

    # train_loader = torch.utils.data.DataLoader(
    #     DeepLesionDataset(
    #         file_list_train, 
    #         return_gt=False,
    #         n_proj=N_PROJ,
    #         img_size=512),
    #     **params
    # )

    # test_loader = torch.utils.data.DataLoader(
    #     DeepLesionDataset(
    #         random.choices(file_list_test, k=1000), 
    #         return_gt=True,
    #         n_proj=N_PROJ,
    #         img_size=512),
    #     **params
    # )

    train_loader = torch.utils.data.DataLoader(
        EllipsesDataset(
            ellipses_type='train', 
            return_gt=False,
            n_proj=N_PROJ,
            img_size=512),
        **params
    )

    test_loader = torch.utils.data.DataLoader(
        EllipsesDataset(
            ellipses_type='validation',
            return_gt=True,
            n_proj=N_PROJ,
            img_size=512),
        **params
    )

    theta = np.linspace(0.0, 180.0, N_PROJ, endpoint=False)
    recon_n2self = N2SelfReconstructor(
        'N2SelfTrained',
        net='unet', lr=0.0001,
        n2self_weights=None,#'selfsuper-ellipses-64-l1-train1/iter_180000.pth',#'iter_15000.pth',
        #'selfsuper-ellipses-64-train8/iter_58800.pth', #'self-super-train9/iter_199800.pth',
        learnable_filter=False
    )
    recon_n2self.init_train(theta)
    recon_n2self._eval(test_loader)

    for i in range(50):
        print('--------------- ',i)
        recon_n2self._train_one_epoch(train_loader, test_loader)
        recon_n2self._eval(test_loader)
        recon_n2self._save('epoch_{}.pth'.format(i))
    recon_n2self._save('end.pth')
