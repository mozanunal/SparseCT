
from sparse_ct.tool import plot_grid
from sparse_ct.data import image_to_sparse_sinogram, ellipses_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
    IRadonReconstructor,
    SartReconstructor,
    SartTVReconstructor,
    SartBM3DReconstructor,
    N2SelfReconstructor,
    SupervisedReconstructor,
    SupervisedItReconstructor
)


def FOCUS2(x):
    return x[200:300, 0:100], (200, 0, 300, 100)


if __name__ == "__main__":

    # fname = "../data/benchmark_ellipses/2.png"
    # fname = "../data/shepp_logan.jpg"
    # fname = "../data/ct1.jpg"
    fname = "../data/benchmark_human/19.png"

    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
                                                          n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=37.0)
    # gt, sinogram, theta, FOCUS = ellipses_to_sparse_sinogram(part='validation', channel=1,
    #         n_proj=64, size=512, angle1=0.0, angle2=180.0, noise_pow=25.0 )

    recons = [
        IRadonReconstructor('FBP'),
        SartReconstructor(
            'SART',
            sart_n_iter=4,
            sart_relaxation=0.15),
        SartTVReconstructor(
            'SART+TV',
            sart_n_iter=4,
            sart_relaxation=0.15,
            tv_weight=0.9,
            tv_n_iter=100),
        SartBM3DReconstructor(
            'SART+BM3D',
            sart_n_iter=4,
            sart_relaxation=0.15,
            bm3d_sigma=0.35),
        # N2SelfReconstructor(
        #     'N2Self',
        #     net='unet',
        #     lr=0.00003,
        #     n2self_weights='iter_187000.pth',
        #     n2self_selfsupervised=True,
        #     learnable_filter=True,
        #     n2self_n_iter=101),
        N2SelfReconstructor(
            'LearnedN2Self',
            net='unet',
            n2self_weights='self-super-train9/iter_199800.pth',
            n2self_selfsupervised=False,
            learnable_filter=False),
        N2SelfReconstructor(
            'LearnedN2SelfHuman',
            net='unet',
            n2self_weights='self-super-human-train-2/iter_68000.pth',
            n2self_selfsupervised=False,
            learnable_filter=False),
        SupervisedReconstructor(
            'FBP+Unet',
            weights='supervised-human-train1/iter_406000.pth',
            net='unet'),
        SupervisedItReconstructor(
            'FBP+Unet+It',
            weights='supervised-human-train1/iter_406000.pth',
            net='unet'),
    ]
    # for metric
    recon_sart_tv = SartTVReconstructor('SART+TV', 
                                sart_n_iter=40, sart_relaxation=0.15,
                                tv_weight=0.9, tv_n_iter=100)
    img_sart_tv = recon_sart_tv.calc(sinogram, theta)

    # reconstruct
    imgs = []
    for recon in recons:
        if hasattr(recon, 'set_for_metric'):
            recon.set_for_metric(gt, img_sart_tv, FOCUS=FOCUS, log_dir='../log/dip')
        if hasattr(recon, 'init_train'):
            recon.init_train(theta)
        imgs.append( recon.calc(sinogram, theta) )

    # metric
    for r in recons:
        mse, psnr, ssim = r.eval(gt)
        print("{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            r.name, mse, psnr, ssim
        ))

    imgs[0] = gt
    plot_grid(imgs, FOCUS=FOCUS, save_name='all1.png',
              dpi=500, plot1d=None, number_of_rows=2)
    plot_grid(imgs, ZOOM=FOCUS, save_name='all2.png',
              dpi=500, plot1d=None, number_of_rows=2)
    plot_grid(imgs, FOCUS=FOCUS2, save_name='all3.png',
              dpi=500, plot1d=None, number_of_rows=2)
    plot_grid(imgs, ZOOM=FOCUS2, save_name='all4.png',
              dpi=500, plot1d=None, number_of_rows=2)
