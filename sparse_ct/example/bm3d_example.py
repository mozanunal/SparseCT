
from skimage.color import gray2rgb
from multiprocessing.dummy import Pool as ThreadPool

from sparse_ct.tool import plot_grid
from sparse_ct.data import image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        FBP_BM3DReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        SartBM3DReconstructor,
                        SartBM3DReconstructor
                        )



if __name__ == "__main__":

    fname = "../data/benchmark_human/19.png"
    # fname = "../data/ct1.jpg"


    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
            n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=30.0 )
            
    n_iter = 40
    sart_relax = 0.15
    recon_fbp = IRadonReconstructor('FBP')
    recons = [
        # SartBM3DReconstructor('BM3D10.00', bm3d_sigma=10.00, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        # SartBM3DReconstructor('BM3D5.00', bm3d_sigma=5.00, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        # SartBM3DReconstructor('BM3D3.00', bm3d_sigma=3.00, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        # SartBM3DReconstructor('BM3D1.50', bm3d_sigma=1.50, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        IRadonReconstructor('FBP'),
        SartReconstructor('SART', sart_n_iter=n_iter, sart_relaxation=sart_relax),
        SartBM3DReconstructor('BM3D1.00', bm3d_sigma=1.00, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        SartBM3DReconstructor('BM3D.95', bm3d_sigma=0.95, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        SartBM3DReconstructor('BM3D.90', bm3d_sigma=0.90, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        SartBM3DReconstructor('BM3D.80', bm3d_sigma=0.80, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        SartBM3DReconstructor('BM3D.50', bm3d_sigma=0.50, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        SartBM3DReconstructor('BM3D.30', bm3d_sigma=0.30, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        SartBM3DReconstructor('BM3D.20', bm3d_sigma=0.20, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        SartBM3DReconstructor('BM3D.10', bm3d_sigma=0.10, sart_n_iter=n_iter, sart_relaxation=sart_relax),
        SartBM3DReconstructor('BM3D.05', bm3d_sigma=0.05, sart_n_iter=n_iter, sart_relaxation=sart_relax),
    ]

    imgs = [ r.calc(sinogram, theta) for r in recons ]
    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    plot_grid( [gt]+imgs, FOCUS=FOCUS, save_name='art2.png', dpi=500, number_of_rows=2  )