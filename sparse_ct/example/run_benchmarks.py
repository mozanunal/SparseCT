
import logging
import numpy as np

from sparse_ct.tool import plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.data.benchmark import benchmark
from sparse_ct.reconstructor_2d import (
    IRadonReconstructor,
    SartReconstructor,
    SartTVReconstructor,
    DgrReconstructor,
    SartBM3DReconstructor)


def benchmark_FBP():
    recon = IRadonReconstructor('FBP', theta)
    theta1 = np.linspace(0.0, 180.0, 100, endpoint=False)
    noise_pow = 25.0
    theta_2 = np.linspace(0.0, 180.0, 32, endpoint=False)





if __name__ == "__main__":

    p_human = '../data/benchmark_human'
    p_ellipses = '../data/benchmark_ellipses'

    

    # recon = SartReconstructor('SART', theta, sart_n_iter=40, sart_relaxation=0.15)

    # recon = SartTVReconstructor('SART+TVw0.9', theta, 
    #                             sart_n_iter=40, sart_relaxation=0.15,
    #                             tv_weight=0.9, tv_n_iter=100)

    # recon = SartBM3DReconstructor('SART+BM3Ds0.35', theta, 
    #                             sart_n_iter=40, sart_relaxation=0.15,
    #                             bm3d_sigma=0.35)

    recon = DgrReconstructor('DGR_1.00_0.00_0.00_0.00', theta, 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=1.0,
                                w_perceptual_loss=0.0,
                                w_tv_loss=0.0
                            )


    recon = DgrReconstructor('RDGR_1.00_0.00_0.00_0.00', theta, 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=1.0,
                                w_perceptual_loss=0.0,
                                w_tv_loss=0.0,
                                randomize_projs=0.1
                            )

    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )

