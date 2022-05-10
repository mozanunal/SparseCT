
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


def benchmark_all(recon, data_list, theta_list, noise_list):
    for data in data_list:
        for theta in theta_list:
            for noise in noise_list:
                benchmark(
                    data,
                    recon,
                    theta,
                    noise
                )



recon_fbp = IRadonReconstructor('FBP')

recon_sart = SartReconstructor('SART', 
                        sart_n_iter=40, 
                        sart_relaxation=0.15)

recon_sart_tv = SartTVReconstructor('SART+TVw0.7',
                        sart_n_iter=40, sart_relaxation=0.15,
                        tv_weight=0.7, tv_n_iter=100)
recon_sart_bm3d = SartBM3DReconstructor('SART+BM3Ds0.20', 
                        sart_n_iter=40, sart_relaxation=0.15,
                        bm3d_sigma=0.20)

recon_dgr = DgrReconstructor('DGR_0.90_0.00_0.00_0.10', 
                        dip_n_iter=4001, 
                        net='skip',
                        lr=0.01,
                        reg_std=1./100,
                        w_proj_loss=0.90,
                        w_perceptual_loss=0.00,
                        w_tv_loss=0.10,
                        w_ssim_loss=0.00)
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

recon_n2s_singleshot = N2SelfReconstructor('N2S_h_L2_train1_128_iter_138000.pth',
                        n2self_weights='selfsuper-human-128-train1/iter_138000.pth',
                        n2self_selfsupervised=False,
                        net='unet',)

recon_learned_supervised = SupervisedReconstructor(
            'SUPERVISED_train1_iter_406000.pth',
            weights='supervised-human-64-train1/iter_406000.pth',
            net='unet')

recon_dgr = DgrReconstructor('DGR_0.00_0.00_0.80_0.20', 
                        dip_n_iter=4001, 
                        net='skip',
                        lr=0.01,
                        reg_std=1./100,
                        w_proj_loss=0.0,
                        w_perceptual_loss=0.00,
                        w_ssim_loss=0.80,
                        w_tv_loss=0.20)


if __name__ == "__main__":

    p_human = '../data/benchmark_human'
    p_ellipses = '../data/benchmark_ellipses'


    data_list = [p_human]
    theta_list = [
                #np.linspace(0.0, 180.0, 32, endpoint=False),
                #np.linspace(0.0, 180.0, 64, endpoint=False),
                np.linspace(0.0, 180.0, 128, endpoint=False),
                ]
    noise_list = [40.0, 37.0, 33.0, 30.0]
    # data_list = [p_ellipses]
    # theta_list = [
    #             np.linspace(0.0, 180.0, 64, endpoint=False),
    #             ]
    # noise_list = [39.0]

    benchmark_all(
        recon_n2s_singleshot,
        data_list,
        theta_list,
        noise_list
    )










