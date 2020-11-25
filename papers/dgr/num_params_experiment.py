
from sparse_ct.tool import plot_grid
from sparse_ct.data import image_to_sparse_sinogram, ellipses_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        SartBM3DReconstructor,
                        DgrReconstructor)



if __name__ == "__main__":
    data_dir = "../../sparse_ct/data/"
    fname = data_dir + "benchmark_ellipses/6.png"

    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
            n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=39.0 )
    recon_sart = SartReconstructor('SART', sart_n_iter=40, sart_relaxation=0.15)
    recon_sart_tv = SartTVReconstructor('SART+TV', 
                                sart_n_iter=40, sart_relaxation=0.15,
                                tv_weight=0.8, tv_n_iter=100)
    recon_dip1 = DgrReconstructor('DGRv1',
                                dip_n_iter=2001, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=0.98,
                                # w_perceptual_loss=0.01,
                                w_tv_loss=0.01,
                                w_ssim_loss=0.01
                            )
    recon_dip2 = DgrReconstructor('DGRv2',
                                dip_n_iter=2001, 
                                net='skipV2',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=0.98,
                                # w_perceptual_loss=0.01,
                                w_tv_loss=0.01,
                                w_ssim_loss=0.01
                            )
    recon_dip3 = DgrReconstructor('DGRv3',
                                dip_n_iter=2001, 
                                net='skipV3',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=0.98,
                                # w_perceptual_loss=0.01,
                                w_tv_loss=0.01,
                                w_ssim_loss=0.01
                            )

    img_sart = recon_sart.calc(sinogram, theta)
    img_sart_tv = recon_sart_tv.calc(sinogram, theta)
    recon_dip1.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log/')
    img_dip1 = recon_dip1.calc(sinogram, theta)
    recon_dip2.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log/')
    img_dip2 = recon_dip2.calc(sinogram, theta)
    recon_dip3.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log/')
    img_dip3 = recon_dip3.calc(sinogram, theta)
    
    recons = [recon_sart, recon_sart_tv, recon_dip1, recon_dip2, recon_dip3,]

    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    plot_grid([
            gt, img_sart, img_sart_tv, img_dip1, img_dip2, img_dip3],
            FOCUS=FOCUS, save_name='num_params.png', dpi=500
        )
            
