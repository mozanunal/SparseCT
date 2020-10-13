
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
    noise_pow = 25.0
    recon = IRadonReconstructor('FBP')
    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )


def benchmark_sart():
    noise_pow = 25.0
    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    recon = SartReconstructor('SART', sart_n_iter=40, sart_relaxation=0.15)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )

def benchmark_sart_tv():
    noise_pow = 25.0
    recon = SartTVReconstructor('SART+TVw1.0',
                                sart_n_iter=40, sart_relaxation=0.15,
                                tv_weight=0.9, tv_n_iter=100)
    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )

def benchmark_sart_bm3d():

    noise_pow = 25.0
    recon = SartBM3DReconstructor('SART+BM3Ds0.35', 
                                sart_n_iter=40, sart_relaxation=0.15,
                                bm3d_sigma=0.35)
    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )

def benchmark_dgr():
    noise_pow = 25.0
    recon = DgrReconstructor('DGR_1.00_0.00_0.00', 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=1.0,
                                w_perceptual_loss=0.0,
                                w_tv_loss=0.0
                            )

    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )


def benchmark_rdgr():
    noise_pow = 25.0

    recon = DgrReconstructor('RDGR_1.00_0.00_0.00', 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=1.0,
                                w_perceptual_loss=0.0,
                                w_tv_loss=0.0,
                                randomize_projs=0.1
                            )
    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        p_human,
        recon,
        theta,
        noise_pow,
    )

if __name__ == "__main__":

    p_human = '../data/benchmark_human'
    p_ellipses = '../data/benchmark_ellipses'

    # benchmark_FBP()
    # benchmark_sart()
    # benchmark_sart_bm3d()
    benchmark_dgr()
    #benchmark_rdgr()
    









