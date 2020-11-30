
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
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("fname")
    args = parser.parse_args()
    print(args.fname)

    #fname = "../../sparse_ct/data/benchmark_ellipses/6.png"
    fname = args.fname # "../../sparse_ct/data/shepp_logan.jpg"
    res_name = fname.split('/')[-1]


    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
            n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=39.0 )

    dgr_iter = 4001
    sart_iter = 40

    recon_fbp = IRadonReconstructor('FBP')
    recon_sart = SartReconstructor('SART', sart_n_iter=sart_iter, sart_relaxation=0.15)
    recon_sart_tv = SartTVReconstructor('SART+TV', 
                                sart_n_iter=sart_iter, sart_relaxation=0.15,
                                tv_weight=0.8, tv_n_iter=100)

    recon_dip1 = DgrReconstructor('DGR1',
                                dip_n_iter=dgr_iter, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=1.00,
                                # w_perceptual_loss=0.01,
                                w_tv_loss=0.0,
                                w_ssim_loss=0.00
                            )
    recon_dip2 = DgrReconstructor('DGR2',
                                dip_n_iter=dgr_iter, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=0.90,
                                # w_perceptual_loss=0.01,
                                w_tv_loss=0.1,
                                w_ssim_loss=0.00
                            )
    recon_dip3 = DgrReconstructor('DGR3', 
                                dip_n_iter=dgr_iter, 
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

    recon_dip1.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log')
    img_dip1 = recon_dip1.calc(sinogram, theta)

    recon_dip2.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log')
    img_dip2 = recon_dip2.calc(sinogram, theta)

    recon_dip3.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log')
    img_dip3 = recon_dip3.calc(sinogram, theta)

    recons = [recon_fbp, recon_sart, recon_sart_tv,
              recon_dip1, recon_dip2, recon_dip3]

    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    for i, img in enumerate([gt, img_fbp, img_sart, img_sart_tv, img_dip1, img_dip2, img_dip3,]):
        plot_grid([img],
            FOCUS=None, save_name=res_name+'_'+str(i)+'_all.png', dpi=500
        )

    plot_grid([
            gt, img_fbp, img_sart, img_sart_tv, 
            img_dip1, img_dip2, img_dip3,],
            FOCUS=None, save_name=res_name+'_all.png', dpi=500
        )
            
