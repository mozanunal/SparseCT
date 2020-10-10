
from skimage.color import gray2rgb

from sparse_ct.tool import plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        SartBM3DReconstructor)



if __name__ == "__main__":

    fname = "../data/ct1.jpg"


    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1, n_proj=32, size=512 )

    n_iter = 40

    recons = [
        IRadonReconstructor('FBP', theta),
        SartReconstructor(
            'SART', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15
        ),
        SartTVReconstructor(
            'SART+TVw0.2', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15, 
            tv_weight=0.2, 
            tv_n_iter=100
        ),
        SartTVReconstructor(
            'SART+TVw0.5', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15, 
            tv_weight=0.5, 
            tv_n_iter=100
        ),
        SartTVReconstructor(
            'SART+TVw0.8', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15, 
            tv_weight=0.8, 
            tv_n_iter=100
        ),
        SartTVReconstructor(
            'SART+TVw0.95', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15, 
            tv_weight=0.95, 
            tv_n_iter=100
        ),
        SartBM3DReconstructor(
            'SART+BM3Dsigma0.10', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15,
            bm3d_sigma=0.10
        ),
        SartBM3DReconstructor(
            'SART+BM3Dsigma0.20', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15,
            bm3d_sigma=0.20
        ),
        SartBM3DReconstructor(
            'SART+BM3Dsigma0.50', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15,
            bm3d_sigma=0.50
        ),
        SartBM3DReconstructor(
            'SART+BM3Dsigma0.70', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15,
            bm3d_sigma=0.70
        ),
        SartBM3DReconstructor(
            'SART+BM3Dsigma0.90', theta, 
            sart_n_iter=n_iter, 
            sart_relaxation=0.15,
            bm3d_sigma=0.90
        ),
    ]
    
    imgs = [ r.calc(sinogram) for r in recons ]

    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    plot_grid( [gt]+imgs, FOCUS=FOCUS, show=True)