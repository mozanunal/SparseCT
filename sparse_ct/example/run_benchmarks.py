
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
    SartBM3DReconstructor,
    N2SelfReconstructor)


def benchmark_FBP(data):
    noise_pow = 25.0
    recon = IRadonReconstructor('FBP')
    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )


def benchmark_sart(data):
    noise_pow = 25.0
    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    recon = SartReconstructor('SART', sart_n_iter=40, sart_relaxation=0.15)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )

def benchmark_sart_tv(data):
    noise_pow = 25.0
    recon = SartTVReconstructor('SART+TVw0.9',
                                sart_n_iter=40, sart_relaxation=0.15,
                                tv_weight=0.9, tv_n_iter=100)
    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )

def benchmark_sart_bm3d(data):

    noise_pow = 25.0
    recon = SartBM3DReconstructor('SART+BM3Ds0.35', 
                                sart_n_iter=40, sart_relaxation=0.15,
                                bm3d_sigma=0.35)
    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )

def benchmark_dgr(data):
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
        data,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )


def benchmark_rdgr(data):
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
        data,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )

def benchmark_n2self(data):
    noise_pow = 25.0

    recon = N2SelfReconstructor('N2Self_SelfSupervised0.2',
                n2self_n_iter=4001, n2self_proj_ratio=0.2,
                n2self_weights=None, n2self_selfsupervised=True,
                net='skipV2', lr=0.01, )


    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )

def benchmark_n2self_learned_singles_shot(data):
    noise_pow = 25.0

    recon = N2SelfReconstructor('N2SelfLearned_NoIt_03ep8',
                n2self_n_iter=4001, n2self_proj_ratio=0.2,
                n2self_weights='training-03/epoch_8.pth',
                n2self_selfsupervised=False,
                net='skipV2', lr=0.01, )

    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )

def benchmark_n2self_learned_selfsupervised(data):
    noise_pow = 25.0

    recon = N2SelfReconstructor('N2SelfLearned_It_03ep8',
                n2self_n_iter=4001, n2self_proj_ratio=0.2,
                n2self_weights='training-03/epoch_8.pth',
                n2self_selfsupervised=True,
                net='skipV2', lr=0.01, )

    theta = np.linspace(0.0, 180.0, 32, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )
    theta = np.linspace(0.0, 180.0, 100, endpoint=False)
    benchmark(
        data,
        recon,
        theta,
        noise_pow,
    )


if __name__ == "__main__":

    p_human = '../data/benchmark_human'
    p_ellipses = '../data/benchmark_ellipses'

    # benchmark_FBP(p_ellipses)
    # benchmark_sart(p_ellipses)
    # benchmark_sart_tv(p_ellipses)
    # benchmark_sart_bm3d(p_ellipses)
    # benchmark_dgr(p_human)
    # benchmark_dgr(p_ellipses)
    # benchmark_rdgr(p_human)
    # benchmark_rdgr(p_ellipses)
    # benchmark_n2self_learned_singles_shot(p_human)
    # benchmark_n2self_learned_singles_shot(p_ellipses)
    # benchmark_n2self(p_human)
    # benchmark_n2self(p_ellipses)
    benchmark_n2self_learned_selfsupervised(p_human)
    benchmark_n2self_learned_selfsupervised(p_ellipses)
    
    









