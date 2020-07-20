
from sparse_ct.tool import plot_result, plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        DipReconstructor)



def test(fname, label):   
    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
            n_proj=32, size=512, angle1=0.0, angle2=180.0 )

    recon_fbp = IRadonReconstructor('FBP', theta)
    recon_sart = SartReconstructor('SART', theta, sart_n_iter=70, sart_relaxation=0.04)
    recon_dip1 = DipReconstructor('DIP_1.00_0.00_0.00_0.00', theta, 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.001,
                                reg_std=1./100,
                                w_proj_loss=1.0,
                                w_perceptual_loss=0.0,
                                w_tv_loss=0.0
                            )
    recon_dip2 = DipReconstructor('DIP_0.00_1.00_0.00_0.00', theta, 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.001,
                                reg_std=1./100,
                                w_proj_loss=0.0,
                                w_perceptual_loss=1.0,
                                w_tv_loss=0.0
                            )
    recon_dip3 = DipReconstructor('DIP_0.50_0.50_0.00_0.00', theta, 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.001,
                                reg_std=1./100,
                                w_proj_loss=0.5,
                                w_perceptual_loss=0.5,
                                w_tv_loss=0.0
                            )

    recon_dip4 = DipReconstructor('DIP_0.99_0.01_0.00_0.00', theta, 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.001,
                                reg_std=1./100,
                                w_proj_loss=0.99,
                                w_perceptual_loss=0.01,
                                w_tv_loss=0.0
                            )


    recon_dip5 = DipReconstructor('DIP_0.50_0.00_0.50_0.00', theta, 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.001,
                                reg_std=1./100,
                                w_proj_loss=0.5,
                                w_perceptual_loss=0.0,
                                w_tv_loss=0.5
                            )
    recon_dip6 = DipReconstructor('DIP_0.90_0.00_0.10_0.00', theta, 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.001,
                                reg_std=1./100,
                                w_proj_loss=0.9,
                                w_perceptual_loss=0.0,
                                w_tv_loss=0.1
                            )
    recon_dip7 = DipReconstructor('DIP_0.99_0.00_0.01_0.00', theta, 
                                dip_n_iter=8000, 
                                net='skip',
                                lr=0.001,
                                reg_std=1./100,
                                w_proj_loss=0.99,
                                w_perceptual_loss=0.0,
                                w_tv_loss=0.01
                            )


    img_fbp = recon_fbp.calc(sinogram)
    img_sart = recon_sart.calc(sinogram)

    recon_dip1.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='../log/dip1')
    img_dip1 = recon_dip1.calc(sinogram)

    recon_dip2.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='../log/dip2')
    img_dip2 = recon_dip2.calc(sinogram)

    recon_dip3.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='../log/dip3')
    img_dip3 = recon_dip3.calc(sinogram)

    recon_dip4.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='../log/dip4')
    img_dip4 = recon_dip4.calc(sinogram)

    recon_dip5.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='../log/dip5')
    img_dip5 = recon_dip5.calc(sinogram)

    recon_dip6.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='../log/dip6')
    img_dip6 = recon_dip6.calc(sinogram)

    recon_dip7.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='../log/dip7')
    img_dip7 = recon_dip7.calc(sinogram)


    for r in [recon_fbp, recon_sart, recon_dip1,
              recon_dip2, recon_dip3, recon_dip4,
              recon_dip5, recon_dip6, recon_dip7]:
        mse, psnr, ssim = r.eval(gt)
        r.save_result()
        print( "{}-> {}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            label, r.name, mse, psnr, ssim
        ))

    plot_grid([gt, img_fbp, img_sart, img_dip1, 
                   img_dip2, img_dip3, img_dip4,
                   img_dip5, img_dip6, img_dip7],
            FOCUS=FOCUS, save_name=label+'.png', dpi=2000)
            

if __name__ == "__main__":
    test("../data/shepp_logan.jpg", "shepp_logan")
    test("../data/ct2.jpg", "ct2")
    test("../data/ct1.jpg", "ct1")
    test("../data/abdomen_ct.jpg", "abdomen_ct")
    test("../data/walnut.jpg", "walnut")