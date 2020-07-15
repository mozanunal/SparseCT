

from sparse_ct.tool import plot_result
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor)



if __name__ == "__main__":

    fname = "../data/ct1.jpg"


    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1, n_proj=32, size=512 )

    recon_fbp = IRadonReconstructor('FBP', theta)
    recon_sart = SartReconstructor('SART', theta, sart_n_iter=40, sart_relaxation=0.07)
    


    plot_result( gt, 
                recon_fbp.calc(sinogram), 
                recon_sart.calc(sinogram,sart_plot=False), 
                FOCUS=FOCUS, show=True)

    for r in [recon_fbp, recon_sart]:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))