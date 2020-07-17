
from skimage.color import gray2rgb

from sparse_ct.tool import plot_result, plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        DipReconstructor)



if __name__ == "__main__":

    fname = "../data/ct1.jpg"


    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1, n_proj=32, size=512 )

    recon_fbp = IRadonReconstructor('FBP', theta)
    recon_sart = SartReconstructor('SART', theta, sart_n_iter=70, sart_relaxation=0.02)
    recon_dip = DipReconstructor('DIP', theta, 
                                dip_n_iter=3000, 
                                net='skip',
                                lr=0.01,
                                reg_std=0.02
                            )
    img_fbp = recon_fbp.calc(sinogram)
    img_sart = recon_sart.calc(sinogram)
    recon_dip.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='../log/dip')
    img_dip = recon_dip.calc(sinogram)




    for r in [recon_fbp, recon_sart, recon_dip]:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    plot_grid([gt, img_fbp, img_sart, img_dip],
            FOCUS=FOCUS, show=True)