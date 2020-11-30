
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
    # file is tested with 4 different noise levels
    # 25, 33, 40 db
    data_dir = "../../sparse_ct/data/"
    fname = data_dir + "benchmark_ellipses/5.png"

    imgs = []
    for dose in [8,16,32,64,128,256,512]:
        print('DOSE:', dose)
        gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
                n_proj=dose, size=512, angle1=0.0, angle2=180.0, noise_pow=39.0 )

        recon_sart = SartReconstructor('SART', sart_n_iter=40, sart_relaxation=0.15)
        recon_dip = DgrReconstructor('DGRM',
                                    dip_n_iter=2501, 
                                    net='skipV2',
                                    lr=0.01,
                                    reg_std=1./100,
                                    w_proj_loss=1.00,
                                    # w_perceptual_loss=0.01,
                                    w_tv_loss=0.00,
                                    w_ssim_loss=0.00
                                )
        recon_dip2 = DgrReconstructor('DGRH',
                                    dip_n_iter=2501, 
                                    net='skipV2',
                                    lr=0.01,
                                    reg_std=1./100,
                                    w_proj_loss=0.98,
                                    # w_perceptual_loss=0.01,
                                    w_tv_loss=0.01,
                                    w_ssim_loss=0.01
                                )

        img_sart = recon_sart.calc(sinogram, theta)
        recon_dip.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log/')
        recon_dip2.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='log/')
        img_dip = recon_dip.calc(sinogram, theta)
        img_dip2 = recon_dip2.calc(sinogram, theta)
        
        recons = [recon_sart, recon_dip, recon_dip2]

        for r in recons:
            mse, psnr, ssim = r.eval(gt)
            print( "{}:{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
                dose, r.name, mse, psnr, ssim
            ))

        plot_grid([img_sart, img_dip, img_dip2],
                FOCUS=FOCUS, save_name='dose'+str(dose)+'.png', dpi=500
            )
        imgs.append(img_dip)

    plot_grid([gt]+imgs,
            FOCUS=FOCUS, save_name='dose.png', dpi=500
        )
            
