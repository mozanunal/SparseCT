
from sparse_ct.tool import plot_result, plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        SartBM3DReconstructor,
                        N2SelfReconstructor
                        )



if __name__ == "__main__":

    fname = "../data/ct1.jpg"


    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
            n_proj=32, size=512, angle1=0.0, angle2=180.0, noise_pow=15.0 )

    recon_fbp = IRadonReconstructor('FBP', theta)
    recon_sart = SartReconstructor('SART', theta, 
                                sart_n_iter=40, sart_relaxation=0.15)
    recon_sart_tv = SartTVReconstructor('SART+TV', theta, 
                                sart_n_iter=40, sart_relaxation=0.15,
                                tv_weight=0.5, tv_n_iter=100)
    recon_bm3d = SartBM3DReconstructor('SART+BM3D', theta, 
                                sart_n_iter=40, sart_relaxation=0.15,
                                bm3d_sigma=0.8)

    recon_n2self = N2SelfReconstructor('N2Self', theta,
                n2self_n_iter=4000, net='skip', lr=0.01)

    img_fbp = recon_fbp.calc(sinogram)
    img_sart = recon_sart.calc(sinogram)
    img_sart_tv = recon_sart_tv.calc(sinogram)
    img_sart_bm3d = recon_bm3d.calc(sinogram)

    recon_n2self.set_for_metric(gt, img_sart_tv, FOCUS=FOCUS, log_dir='../log/dip')
    img_n2self = recon_n2self.calc(sinogram)


    recons = [recon_fbp, recon_sart, 
              recon_sart_tv, recon_bm3d,
              recon_n2self]

    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    plot_grid([gt, img_fbp, img_sart, img_sart_tv, img_sart_bm3d, img_n2self],
            FOCUS=FOCUS, save_name='all.png', dpi=500)
            
