
from skimage.color import gray2rgb
from multiprocessing.dummy import Pool as ThreadPool

from sparse_ct.tool import plot_grid
from sparse_ct.data import image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        SartBM3DReconstructor)



if __name__ == "__main__":

    fname = "../data/benchmark_human/20.png"


    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1, n_proj=32, size=512, noise_pow=33.0 )

    n_iter = 40
    sart_relax = 0.15
    recon_fbp = IRadonReconstructor('FBP')
    recons = [
        SartBM3DReconstructor('BM3D95', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.95),
        SartBM3DReconstructor('BM3D90', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.90),
        SartBM3DReconstructor('BM3D80', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.80),
        SartBM3DReconstructor('BM3D50', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.50),
        SartBM3DReconstructor('BM3D30', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.30),
        SartBM3DReconstructor('BM3D15', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.15),
        SartBM3DReconstructor('BM3D07', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.07),
        SartBM3DReconstructor('BM3D05', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.05),
        SartBM3DReconstructor('BM3D03', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.03),
        SartBM3DReconstructor('BM3D02', sart_n_iter=n_iter, sart_relaxation=sart_relax, bm3d_sigma=0.02),
    ]

    imgs = [ r.calc(sinogram, theta) for r in recons ]
    pool = ThreadPool(5)
    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    plot_grid( [gt]+imgs, FOCUS=FOCUS, save_name='art.png', dpi=500 )