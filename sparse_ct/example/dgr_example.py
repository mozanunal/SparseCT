
from sparse_ct.tool import plot_grid
from sparse_ct.data import sparse_image, image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import (
    IRadonReconstructor,
    SartReconstructor,
    SartTVReconstructor,
    DgrReconstructor)


def test(fname, label):
    gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname,
                                                          channel=1, n_proj=32, size=512,
                                                          angle1=0.0, angle2=180.0, noise_pow=15.0)

    recon_fbp = IRadonReconstructor('FBP', theta)
    recon_sart = SartReconstructor(
        'SART', theta,
        sart_n_iter=70,
        sart_relaxation=0.04)

    recon_dips = [
        DgrReconstructor('DIP_1.00_0.00_0.00_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=1.0,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.99_0.01_0.00_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.99,
                         w_perceptual_loss=0.01,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.90_0.10_0.00_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.90,
                         w_perceptual_loss=0.10,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.50_0.50_0.00_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.5,
                         w_perceptual_loss=0.5,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.10_0.90_0.00_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.10,
                         w_perceptual_loss=0.90,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.01_0.99_0.00_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.01,
                         w_perceptual_loss=0.99,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.00_1.00_0.00_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.0,
                         w_perceptual_loss=1.0,
                         w_tv_loss=0.0
                         ),
        DgrReconstructor('DIP_0.99_0.00_0.01_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.99,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.01
                         ),
        DgrReconstructor('DIP_0.90_0.00_0.10_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.9,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.1
                         ),
        DgrReconstructor('DIP_0.50_0.00_0.50_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.5,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.5
                         ),
        DgrReconstructor('DIP_0.10_0.00_0.90_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.1,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.9
                         ),
        DgrReconstructor('DIP_0.01_0.00_0.99_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.01,
                         w_perceptual_loss=0.0,
                         w_tv_loss=0.99
                         ),
        DgrReconstructor('DIP_0.00_0.00_1.0_0.00', theta,
                         dip_n_iter=8000,
                         net='skip',
                         lr=0.001,
                         reg_std=1./100,
                         w_proj_loss=0.00,
                         w_perceptual_loss=0.0,
                         w_tv_loss=1.0
                         ),
    ]

    img_fbp = recon_fbp.calc(sinogram)
    img_sart = recon_sart.calc(sinogram)

    imgs = []
    for recon in recon_dips:
        recon.set_for_metric(gt, img_sart, FOCUS=FOCUS, log_dir='../log/dip1')
        imgs.append(recon.calc(sinogram))

    for r in [recon_fbp, recon_sart] + recon_dips:
        mse, psnr, ssim = r.eval(gt)
        r.save_result()
        print("{}-> {}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            label, r.name, mse, psnr, ssim
        ))

    plot_grid([gt, img_fbp, img_sart] + imgs,
              FOCUS=FOCUS, save_name=label+'.png', dpi=500)


if __name__ == "__main__":
    # test("../data/shepp_logan.jpg", "shepp_logan")
    # test("../data/ct2.jpg", "ct2")
    # test("../data/ct1.jpg", "ct1")
    # test("../data/LoDoPaB/004013_02_01_119.png", "LoDoPaB1")
    # test("../data/LoDoPaB/004017_01_01_151.png", "LoDoPaB2")
    # test("../data/LoDoPaB/004028_01_04_109.png", "LoDoPaB3")
    test("../data/LoDoPaB/004043_01_01_169.png", "LoDoPaB4")
    test("../data/LoDoPaB/004049_04_01_062.png", "LoDoPaB5")

