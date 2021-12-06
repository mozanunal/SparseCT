
import logging
import numpy as np

from sparse_ct.tool import plot_grid
from sparse_ct.data import image_to_sparse_sinogram
from sparse_ct.data.benchmark import benchmark
from sparse_ct.reconstructor_2d import (
    IRadonReconstructor,
    SartReconstructor,
    SartTVReconstructor,
    DgrReconstructor,
    SartBM3DReconstructor,
    N2SelfReconstructor,
    SupervisedReconstructor)


def benchmark_all(recon, data_list, theta_list):
    for data in data_list:
        for theta in theta_list:
            benchmark(
                data,
                recon,
                theta,
                40.0
            )



recon_fbp = IRadonReconstructor('FBP')

recon_sart = SartReconstructor('SART', 
                        sart_n_iter=40, 
                        sart_relaxation=0.15)

recon_sart_tv = SartTVReconstructor('SART+TVw0.9',
                        sart_n_iter=40, sart_relaxation=0.15,
                        tv_weight=0.9, tv_n_iter=100)
recon_sart_bm3d = SartBM3DReconstructor('SART+BM3Ds0.35', 
                        sart_n_iter=40, sart_relaxation=0.15,
                        bm3d_sigma=0.35)

recon_dgr = DgrReconstructor('DGR_0.80_0.00_0.10_0.10', 
                        dip_n_iter=4001, 
                        net='skip',
                        lr=0.01,
                        reg_std=1./100,
                        w_proj_loss=0.80,
                        w_perceptual_loss=0.00,
                        w_tv_loss=0.10,
                        w_ssim_loss=0.10)
recon_rdgr = DgrReconstructor('RDGR_1.00_0.00_0.00', 
                        dip_n_iter=4001, 
                        net='skip',
                        lr=0.01,
                        reg_std=1./100,
                        w_proj_loss=1.0,
                        w_perceptual_loss=0.0,
                        w_tv_loss=0.0,
                        randomize_projs=0.1)

recon_n2s_selfsuper = N2SelfReconstructor('N2S_SelfSup_02',
                        n2self_n_iter=4001,
                        n2self_weights=None, 
                        n2self_selfsupervised=True,
                        net='skip', lr=0.01, )

recon_n2s_singleshot = N2SelfReconstructor('N2S_SingleS-L2',
                        n2self_weights='selfsuper-ellipses-64-train9/iter_199800.pth',
                        n2self_selfsupervised=False,
                        net='unet',)

recon_learned_selfsuper = N2SelfReconstructor('N2S_LearSelfSup_02_05',
                        n2self_n_iter=4001, n2self_proj_ratio=0.2,
                        n2self_weights='iter_95000.pth',#'training-04/iter_100000.pth',
                        n2self_selfsupervised=True,
                        net='skipV2', lr=0.01, )

recon_learned_supervised = SupervisedReconstructor(
            'FBP+Unet+Human',
            weights='supervised-human-64-train1/iter_406000.pth',
            net='unet')


if __name__ == "__main__":

    p_human = '../data/benchmark_human'
    p_ellipses = '../data/benchmark_ellipses'


    data_list = [p_ellipses, p_human]
    theta_list = [
                np.linspace(0.0, 180.0, 32, endpoint=False),
                np.linspace(0.0, 180.0, 64, endpoint=False),
                np.linspace(0.0, 180.0, 100, endpoint=False),
                ]

    benchmark_all(
        recon_learned_supervised,
        data_list,
        theta_list
    )










