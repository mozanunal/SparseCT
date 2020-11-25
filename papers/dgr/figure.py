
from sparse_ct.tool import plot_grid
from sparse_ct.data import image_to_sparse_sinogram, ellipses_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        SartBM3DReconstructor,
                        DgrReconstructor,
                        N2SelfReconstructor)



if __name__ == "__main__":
    #fname = "../../sparse_ct/data/benchmark_ellipses/6.png"
    fname = "../../sparse_ct/data/shepp_logan.jpg"


    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
            n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=39.0 )

    recon_fbp = IRadonReconstructor('FBP')
    recon_sart = SartReconstructor('SART', sart_n_iter=40, sart_relaxation=0.15)
    recon_sart_tv = SartTVReconstructor('SART+TV', 
                                sart_n_iter=40, sart_relaxation=0.15,
                                tv_weight=0.8, tv_n_iter=100)
    recon_bm3d = SartBM3DReconstructor('SART+BM3D',
                                sart_n_iter=40, sart_relaxation=0.15,
                                bm3d_sigma=0.3)

    recon_dip = DgrReconstructor('DGR',
                                dip_n_iter=1501, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=1.00,
                                # w_perceptual_loss=0.01,
                                w_tv_loss=0.0,
                                w_ssim_loss=0.00
                            )
    recon_dip_rand = DgrReconstructor('DGR', 
                                dip_n_iter=1501, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=0.98,
                                # w_perceptual_loss=0.1,
                                w_tv_loss=0.01,
                                w_ssim_loss=0.01
                            )

    img_fbp = recon_fbp.calc(sinogram, theta)
    img_sart = recon_sart.calc(sinogram, theta)
    img_sart_tv = recon_sart_tv.calc(sinogram, theta)
    img_sart_bm3d = recon_bm3d.calc(sinogram, theta)

    recon_dip.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log')
    img_dip = recon_dip.calc(sinogram, theta)

    recon_dip_rand.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log')
    img_dip_rand = recon_dip_rand.calc(sinogram, theta)

    recons = [recon_fbp, recon_sart, 
              recon_sart_tv, recon_bm3d,
              recon_dip, recon_dip_rand,]

    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    for i, img in enumerate([gt, img_fbp, img_sart, img_sart_tv, img_sart_bm3d, img_dip, img_dip_rand,]):
        plot_grid([img],
            FOCUS=None, save_name=str(i)+'_all.png', dpi=500
        )

    plot_grid([
            gt, img_fbp, img_sart, img_sart_tv, img_sart_bm3d, 
            img_dip, img_dip_rand,],
            FOCUS=None, save_name='all.png', dpi=500
        )
            
