
from skimage.color import gray2rgb

from sparse_ct.tool import plot_result, plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor)



if __name__ == "__main__":

    fname = "../data/ct1.jpg"


    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1, n_proj=32, size=512 )

    n_iter = 40
    recon_fbp = IRadonReconstructor('FBP', theta)
    recon_sart_95 = SartReconstructor('SART95', theta, sart_n_iter=n_iter, sart_relaxation=0.95)
    recon_sart_90 = SartReconstructor('SART90', theta, sart_n_iter=n_iter, sart_relaxation=0.90)
    recon_sart_80 = SartReconstructor('SART80', theta, sart_n_iter=n_iter, sart_relaxation=0.80)
    recon_sart_50 = SartReconstructor('SART50', theta, sart_n_iter=n_iter, sart_relaxation=0.50)
    recon_sart_30 = SartReconstructor('SART30', theta, sart_n_iter=n_iter, sart_relaxation=0.30)
    recon_sart_15 = SartReconstructor('SART15', theta, sart_n_iter=n_iter, sart_relaxation=0.15)
    recon_sart_07 = SartReconstructor('SART07', theta, sart_n_iter=n_iter, sart_relaxation=0.07)
    recon_sart_05 = SartReconstructor('SART05', theta, sart_n_iter=n_iter, sart_relaxation=0.05)
    recon_sart_03 = SartReconstructor('SART03', theta, sart_n_iter=n_iter, sart_relaxation=0.03)
    recon_sart_02 = SartReconstructor('SART02', theta, sart_n_iter=n_iter, sart_relaxation=0.02)

    recons = [
        recon_fbp,
        recon_sart_95,
        recon_sart_90,
        recon_sart_80,
        recon_sart_50,
        recon_sart_30,
        recon_sart_15,
        recon_sart_07,
        recon_sart_05,
        recon_sart_03,
        recon_sart_02,
    ]
    
    imgs = [ r.calc(sinogram) for r in recons ]

    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    plot_grid( [gt]+imgs, FOCUS=FOCUS, show=True)