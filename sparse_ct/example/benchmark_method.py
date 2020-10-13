
import logging
import numpy as np

from sparse_ct.tool import plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
    IRadonReconstructor,
    SartReconstructor,
    SartTVReconstructor,
    DgrReconstructor,
    SartBM3DReconstructor)



 


def benchmark(fname, label, n_proj=32, noise_pow=25.0):

    # data
    n_proj = 32
    noise_pow = 25.0
    images = [
        "../data/shepp_logan.jpg",
        "../data/ct2.jpg",
        "../data/ct1.jpg",
        "../data/LoDoPaB/004013_02_01_119.png",
        "../data/LoDoPaB/004017_01_01_151.png",
        "../data/LoDoPaB/004028_01_04_109.png",
        "../data/LoDoPaB/004043_01_01_169.png",
        "../data/LoDoPaB/004049_04_01_062.png",
    ]

    theta = np.linspace(0.0, 180.0, n_proj, endpoint=False)



    # method
    dgr_iter = 4000
    lr = 0.01
    net = 'skip'
    noise_std = 1./100

    recon = DgrReconstructor('DIP_1.00_0.00_0.00_0.00', theta,
                    dip_n_iter=dgr_iter,
                    net=net,
                    lr=lr,
                    reg_std=noise_std,
                    w_proj_loss=1.0,
                    w_perceptual_loss=0.0,
                    w_tv_loss=0.0
                )

    # run
    logging.basicConfig(
        filename='{}.log'.format(recon.name, ), 
        filemode='a', 
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.DEBUG
    )

    logging.warning('Starting')
    logging.warning('fname: %s %s',label, fname)
    logging.warning('n_proj: %s', n_proj)
    logging.warning('noise_pow: %s', noise_pow)
    logging.warning('dgr_n_iter: %s', dgr_iter)
    logging.warning('dgr_lr: %s', lr)
    logging.warning('dgr_net: %s', net)
    logging.warning('dgr_noise_std: %s', noise_std)



    for fname in images:
        gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname,
                            channel=1, n_proj=n_proj, size=512,
                            angle1=0.0, angle2=180.0, noise_pow=noise_pow)
        # set metrics
        if type(recon) == DgrReconstructor:
            recon_bm3d = SartBM3DReconstructor('SART+BM3D', theta, 
                            sart_n_iter=40, sart_relaxation=0.15,
                            bm3d_sigma=0.5)
            img_sart_bm3d = recon_bm3d.calc(sinogram)
            recon.set_for_metric(gt, img_sart_bm3d, FOCUS=FOCUS, log_dir='../log/dip')

        recon.calc(sinogram)
        mse, psnr, ssim = recon.eval(gt)
        logstr = "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            recon.name, mse, psnr, ssim
        )
        logging.info(logstr)

    logging.warning('Done. Results saved as %s')

if __name__ == "__main__":
    # 
    test("../data/shepp_logan.jpg", "shepp_logan_32_35", n_proj=32, noise_pow=35.0)
    test("../data/ct2.jpg", "ct2_32_35", n_proj=32, noise_pow=35.0)
    test("../data/ct1.jpg", "ct1_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004013_02_01_119.png", "LoDoPaB1_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004017_01_01_151.png", "LoDoPaB2_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004028_01_04_109.png", "LoDoPaB3_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004043_01_01_169.png", "LoDoPaB4_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004049_04_01_062.png", "LoDoPaB5_32_35", n_proj=32, noise_pow=35.0)

