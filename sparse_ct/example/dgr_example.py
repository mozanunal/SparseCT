
from sparse_ct.tool import plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
    IRadonReconstructor,
    SartReconstructor,
    SartTVReconstructor,
    DgrReconstructor,
    SartBM3DReconstructor)

import logging

logging.basicConfig(
    filename='dgr_example_32_35.log', 
    filemode='a', 
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.DEBUG
)
 


def test(fname, label, n_proj=32, noise_pow=25.0):

    
    dgr_iter = 4000
    lr = 0.01
    net = 'skip'
    noise_std = 1./100

    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname,
                                    channel=1, n_proj=n_proj, size=512,
                                    angle1=0.0, angle2=180.0, noise_pow=noise_pow)

    logging.warning('Starting')
    logging.warning('fname: %s %s',label, fname)
    logging.warning('n_proj: %s', n_proj)
    logging.warning('noise_pow: %s', noise_pow)
    logging.warning('dgr_n_iter: %s', dgr_iter)
    logging.warning('dgr_lr: %s', lr)
    logging.warning('dgr_net: %s', net)
    logging.warning('dgr_noise_std: %s', noise_std)

    recons = [
        IRadonReconstructor('FBP'),
        SartReconstructor('SART', sart_n_iter=40, sart_relaxation=0.15),
        SartTVReconstructor('SART+TV', 
                                    sart_n_iter=40, sart_relaxation=0.15,
                                    tv_weight=0.5, tv_n_iter=100),
        SartBM3DReconstructor('SART+BM3D', 
                                    sart_n_iter=40, sart_relaxation=0.15,
                                    bm3d_sigma=0.5),    
        DgrReconstructor('DIP_1.00_0.00_0.00_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=1.0,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.99_0.01_0.00_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.99,
                         w_perceptual_loss=0.01,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.90_0.10_0.00_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.90,
                         w_perceptual_loss=0.10,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.50_0.50_0.00_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.5,
                         w_perceptual_loss=0.5,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.10_0.90_0.00_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.10,
                         w_perceptual_loss=0.90,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.01_0.99_0.00_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.01,
                         w_perceptual_loss=0.99,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.00_1.00_0.00_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.0,
                         w_perceptual_loss=1.0,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.99_0.00_0.01_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.99,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.01
                         ),
        DgrReconstructor('DIP_0.90_0.00_0.10_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.9,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.1
                         ),
        DgrReconstructor('DIP_0.50_0.00_0.50_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.5,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.5
                         ),
        DgrReconstructor('DIP_0.10_0.00_0.90_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.1,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.9
                         ),
        DgrReconstructor('DIP_0.01_0.00_0.99_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.01,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.99
                         ),
        DgrReconstructor('DIP_0.00_0.00_1.0_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.00,
                         w_perceptual_loss=0.0,
                         w_tv_loss=1.0
                         ),



        DgrReconstructor('DIP_0.33_0.33_0.33_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.33,
                         w_perceptual_loss=0.33,
                         w_tv_loss=0.33
                         ),
        DgrReconstructor('DIP_0.8_0.10_0.10_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.8,
                         w_perceptual_loss=0.1,
                         w_tv_loss=0.1
                         ),
        DgrReconstructor('DIP_0.98_0.01_0.01_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.98,
                         w_perceptual_loss=0.01,
                         w_tv_loss=0.01
                         ),

        DgrReconstructor('DIP_0.10_0.80_0.10_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.10,
                         w_perceptual_loss=0.80,
                         w_tv_loss=0.10
                         ),
        DgrReconstructor('DIP_0.01_0.98_0.01_0.00',
                         dip_n_iter=dgr_iter,
                         net=net,
                         lr=lr,
                         reg_std=noise_std,
                         w_proj_loss=0.01,
                         w_perceptual_loss=0.98,
                         w_tv_loss=0.01
                         ),

    ]

    img_sart_bm3d = recons[3].calc(sinogram, theta)

    imgs = []
    for recon in recons:
        if type(recon) == DgrReconstructor:
            recon.set_for_metric(gt, img_sart_bm3d, FOCUS=FOCUS, log_dir='../log/dip')
        imgs.append(recon.calc(sinogram))
        mse, psnr, ssim = recon.eval(gt)
        recon.save_result()
        logstr = "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            recon.name, mse, psnr, ssim
        )
        logging.info(logstr)

    plot_grid([gt] + imgs,
              FOCUS=FOCUS, save_name=label+'.png', dpi=500)

    logging.warning('Done. Results saved as %s', label+'.png')

if __name__ == "__main__":
    # 
    test("../data/shepp_logan.jpg", "shepp_logan_32_35", n_proj=32, noise_pow=35.0)
    test("../data/ct2.jpg", "ct2_32_35", n_proj=32, noise_pow=35.0)
    test("../data/ct1.jpg", "ct1_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004013_02_01_119.png", "LoDoPaB1_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004017_01_01_151.png", "LoDoPaB2_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004028_01_04_109.png", "LoDoPaB3_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004043_01_01_169.png", "LoDoPaB4_32_35", n_proj=32, noise_pow=35.0)
    test("../data/LoDoPaB/004049_04_01_062.png", "LoDoPaB5_32_35", n_proj=32, noise_pow=35.0)

