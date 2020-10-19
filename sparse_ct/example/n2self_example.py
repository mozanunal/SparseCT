
from sparse_ct.tool import plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram, ellipses_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        SartBM3DReconstructor,
                        N2SelfReconstructor
                        )



if __name__ == "__main__":

    fname = "../data/benchmark_human/shepp_logan.jpg"
    #fname = "../data/benchmark_human/ct1.jpg"



    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
            n_proj=128, size=512, angle1=0.0, angle2=180.0, noise_pow=25.0 )
    # gt, sinogram, theta, FOCUS = ellipses_to_sparse_sinogram(part='validation', channel=1,
    #         n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=25.0 )

    recon_fbp = IRadonReconstructor('FBP')
    recon_sart = SartReconstructor('SART',
                                sart_n_iter=40, sart_relaxation=0.15)
    recon_sart_tv = SartTVReconstructor('SART+TV', 
                                sart_n_iter=40, sart_relaxation=0.15,
                                tv_weight=0.9, tv_n_iter=100)
    recon_bm3d = SartBM3DReconstructor('SART+BM3D', 
                                sart_n_iter=40, sart_relaxation=0.15,
                                bm3d_sigma=0.35)

    recon_n2self_selfsuper = N2SelfReconstructor('N2Self_SelfSupervised',
                n2self_n_iter=4001, n2self_proj_ratio=0.05,
                n2self_weights=None, n2self_selfsupervised=True,
                net='skipV2', lr=0.01, )

    recon_n2self_learned_single = N2SelfReconstructor('N2Self_Learned_SingleShot',
                n2self_n_iter=4001, n2self_proj_ratio=0.05,
                n2self_weights='iter_96000.pth',
                n2self_selfsupervised=False,
                net='skipV2', lr=0.01, )

    recon_n2self_learned_selfsuper = N2SelfReconstructor('N2Self_Learned_SelfSupervised',
                n2self_n_iter=4001, n2self_proj_ratio=0.05,
                n2self_weights='iter_96000.pth',
                n2self_selfsupervised=True,
                net='skipV2', lr=0.01, )

    img_fbp = recon_fbp.calc(sinogram, theta)
    img_sart = recon_sart.calc(sinogram, theta)
    img_sart_tv = recon_sart_tv.calc(sinogram, theta)
    img_sart_bm3d = recon_bm3d.calc(sinogram, theta)

    recon_n2self_selfsuper.set_for_metric(gt, img_sart_tv, FOCUS=FOCUS, log_dir='../log/dip')
    img_n2self_selfsuper = recon_n2self_selfsuper.calc(sinogram, theta)

    recon_n2self_learned_single.set_for_metric(gt, img_sart_tv, FOCUS=FOCUS, log_dir='../log/dip')
    img_n2self_learned_single = recon_n2self_learned_single.calc(sinogram, theta)

    recon_n2self_learned_selfsuper.set_for_metric(gt, img_sart_tv, FOCUS=FOCUS, log_dir='../log/dip')
    img_n2self_learned_selfsuper = recon_n2self_learned_selfsuper.calc(sinogram, theta)


    recons = [recon_fbp, recon_sart, 
              recon_sart_tv, recon_bm3d,
              recon_n2self_selfsuper, 
              recon_n2self_learned_single,
              recon_n2self_learned_selfsuper]

    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    plot_grid([gt, img_fbp, img_sart, img_sart_tv, img_sart_bm3d, img_n2self_selfsuper, img_n2self_learned_single, img_n2self_learned_selfsuper],
            FOCUS=None, save_name='all.png', dpi=500)

    plot_grid([gt, img_fbp, img_sart, img_sart_tv, img_sart_bm3d, img_n2self_learned_selfsuper],
            FOCUS=None, save_name='all2.png', dpi=500)
            
