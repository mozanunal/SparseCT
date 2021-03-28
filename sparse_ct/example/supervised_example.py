
from sparse_ct.tool import plot_grid
from sparse_ct.data import image_to_sparse_sinogram, ellipses_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
                        IRadonReconstructor,
                        SartReconstructor,
                        SartTVReconstructor,
                        SartBM3DReconstructor,
                        N2SelfReconstructor,
                        SupervisedReconstructor
                        )

def FOCUS2(x):
    return x[200:300, 0:100], (200, 0, 300, 100)

if __name__ == "__main__":

    # fname = "../data/benchmark_ellipses/2.png"
    # fname = "../data/shepp_logan.jpg"
    # fname = "../data/ct1.jpg"
    fname = "../data/benchmark_human/19.png"



    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
            n_proj=128, size=512, angle1=0.0, angle2=180.0, noise_pow=33.0 )
    # gt, sinogram, theta, FOCUS = ellipses_to_sparse_sinogram(part='validation', channel=1,
    #         n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=25.0 )

    recon_fbp = IRadonReconstructor('FBP')
    recon_sart = SartReconstructor(
                        'SART',
                        sart_n_iter=40, 
                        sart_relaxation=0.15)
    recon_sart_tv = SartTVReconstructor(
                        'SART+TV', 
                        sart_n_iter=40, 
                        sart_relaxation=0.15,
                        tv_weight=0.9, 
                        tv_n_iter=100)
    recon_bm3d = SartBM3DReconstructor(
                        'SART+BM3D', 
                        sart_n_iter=40, 
                        sart_relaxation=0.15,
                        bm3d_sigma=0.35)

    recon_selfsuper = N2SelfReconstructor(
                        'N2Self',
                        net='unet',
                        lr=0.00003,
                        n2self_weights='iter_187000.pth',
                        n2self_selfsupervised=True,
                        learnable_filter=True,
                        n2self_n_iter=101
                    )

    recon_selfsuper_learned = N2SelfReconstructor(
                        'LearnedN2Self',
                        net='unet',
                        n2self_weights='self-super-train9/iter_199800.pth',
                        n2self_selfsupervised=False,
                        learnable_filter=False
                    )
    recon_selfsuper_learned_human = N2SelfReconstructor(
                        'LearnedN2SelfHuman',
                        net='unet',
                        n2self_weights='self-super-human-train-2/iter_68000.pth',
                        n2self_selfsupervised=False,
                        learnable_filter=False
                    )
    
    recon_supervised = SupervisedReconstructor(
                        'FBP+Unet',
                        weights='supervised-human-train1/iter_406000.pth',
                        net='unet')
    


    img_fbp = recon_fbp.calc(sinogram, theta)
    img_sart = recon_sart.calc(sinogram, theta)
    img_sart_tv = recon_sart_tv.calc(sinogram, theta)
    img_sart_bm3d = recon_bm3d.calc(sinogram, theta)

    recon_selfsuper.set_for_metric(gt, img_sart_tv, FOCUS=FOCUS, log_dir='../log/dip')
    recon_selfsuper.init_train(theta)
    img_selfsupervised = recon_selfsuper.calc(sinogram, theta)

    recon_selfsuper_learned.set_for_metric(gt, img_sart_tv, FOCUS=FOCUS, log_dir='../log/dip')
    recon_selfsuper_learned.init_train(theta)
    img_learned_selfsupervised = recon_selfsuper_learned.calc(sinogram, theta)

    recon_selfsuper_learned_human.set_for_metric(gt, img_sart_tv, FOCUS=FOCUS, log_dir='../log/dip')
    recon_selfsuper_learned_human.init_train(theta)
    img_learned_selfsupervised_human = recon_selfsuper_learned_human.calc(sinogram, theta)

    recon_supervised.set_for_metric(gt, img_sart_tv, FOCUS=FOCUS, log_dir='../log/dip')
    recon_supervised.init_train(theta)
    img_supervised = recon_supervised.calc(sinogram, theta)

    recons = [recon_fbp, recon_sart, 
              recon_sart_tv, recon_bm3d,
              recon_selfsuper,
              recon_selfsuper_learned,
              recon_selfsuper_learned_human,
              recon_supervised]

    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print( "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    imgs = [gt, img_fbp, img_sart, img_sart_tv, img_sart_bm3d, img_learned_selfsupervised, img_learned_selfsupervised_human, img_supervised]
    plot_grid(imgs, FOCUS=FOCUS, save_name='all1.png', dpi=500, plot1d=None, number_of_rows=2)            
    plot_grid(imgs, ZOOM=FOCUS, save_name='all2.png', dpi=500, plot1d=None, number_of_rows=2)
    plot_grid(imgs, FOCUS=FOCUS2, save_name='all3.png', dpi=500, plot1d=None, number_of_rows=2)            
    plot_grid(imgs, ZOOM=FOCUS2, save_name='all4.png', dpi=500, plot1d=None, number_of_rows=2)