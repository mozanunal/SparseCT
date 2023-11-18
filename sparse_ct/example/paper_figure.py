
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
    #fname = "../data/benchmark_ellipses/6.png"

    # gt, sinogram, theta, _ = image_to_sparse_sinogram("../data/sl.png", channel=1,
    #         n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=37.0 )
    # def FOCUS(x):
    #     return x[220:300, 200:300], (220, 200, 300, 300)
    
    gt, sinogram, theta, _ = image_to_sparse_sinogram("../data/benchmark_ellipses/7.png", channel=1,
            n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=33.0 )
    def FOCUS(x):
        return x[0:100, 270:420], (0, 270, 100, 420)

    # gt, sinogram, theta, _ = image_to_sparse_sinogram("../data/selected/3.x.png", channel=1,
    #         n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=33.0 )
    # def FOCUS(x):
    #     return x[200:320, 210:310], (200, 210, 320, 310)
    
    # gt, sinogram, theta, _ = image_to_sparse_sinogram("../data/benchmark_human/19.png", channel=1,
    #         n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=37.0 )
    # def FOCUS(x):
    #     return x[200:300, 200:300], (200, 200, 300, 300)
   
    recon_fbp = IRadonReconstructor('FBP')
    recon_sart = SartReconstructor('SART', sart_n_iter=40, sart_relaxation=0.15)
    recon_bm3d = SartBM3DReconstructor('SART+BM3D',
                                sart_n_iter=40, sart_relaxation=0.15,
                                bm3d_sigma=0.2)

    recon_dip = DgrReconstructor('DGR',
                                dip_n_iter=4001, 
                                net='skip',
                                lr=0.01,
                                reg_std=1./100,
                                w_proj_loss=0.95,
                                # w_perceptual_loss=0.01,
                                w_tv_loss=0.05,
                                w_ssim_loss=0.00
                            )
    # img_fbp = recon_fbp.calc(sinogram, theta)
    # img_sart = recon_sart.calc(sinogram, theta)
    # # img_sart_tv = recon_sart_tv.calc(sinogram, theta)
    img_sart_bm3d = recon_bm3d.calc(sinogram, theta)

    # recon_dip.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='./log/dip')
    # img_dip = recon_dip.calc(sinogram, theta)

    for r in [recon_bm3d]:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    plot_grid([img_sart_bm3d], FOCUS=FOCUS, save_name='paper.png', dpi=500)
            
