
import math
import logging
import glob
import numpy as np

from sparse_ct.reconstructor_2d import (
                                SartReconstructor,
                                SartBM3DReconstructor,
                                DgrReconstructor,
                                N2SelfReconstructor)
from . import image_to_sparse_sinogram


def get_images(path):
    img_png = glob.glob(path+'/*.png')
    img_jpg = glob.glob(path+'/*.jpg')
    return img_jpg + img_png

def benchmark(
        images_path,
        recon,
        theta,
        noise_pow
    ):

    images=get_images(images_path)

    log_filename = 'benchmark_{recon}'.format(
        recon=recon.name
    )
    
    logging.basicConfig(
        filename='{}.log'.format(log_filename), 
        filemode='a', 
        format='%(asctime)s - %(levelname)s - %(message)s',
        level=logging.INFO
    )

    logging.warning('Starting')
    logging.warning('images: %s', images_path)
    logging.warning('n_proj: %s', len(theta))
    logging.warning('noise_pow: %s', noise_pow)

    mse_list = []
    psnr_list = []
    ssim_list = []

    for fname in images:
        gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname,
                        channel=1, n_proj=len(theta), size=512,
                        angle1=0.0, angle2=180.0, noise_pow=noise_pow)
        # set metrics
        if type(recon) == DgrReconstructor or type(recon) == N2SelfReconstructor:
            recon_bm3d = SartReconstructor('SART', 
                            sart_n_iter=40, sart_relaxation=0.15)
            img_sart_bm3d = recon_bm3d.calc(sinogram, theta)
            recon.set_for_metric(gt, img_sart_bm3d, FOCUS=FOCUS, log_dir='../log/dip')

        recon.calc(sinogram, theta)
        mse, psnr, ssim = recon.eval(gt)
        mse_list.append(mse)
        psnr_list.append(psnr)
        ssim_list.append(ssim)
        logstr = "{}: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}".format(
            fname, mse, psnr, ssim
        )
        logging.info(logstr)


    logging.info('Avg: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}'.format(
            np.mean(mse_list), 
            np.mean(psnr_list), 
            np.mean(ssim_list)
        ))
    logging.info('Std: MSE:{:.5f} PSNR:{:.5f} SSIM:{:.5f}'.format(
            np.std(mse_list), 
            np.std(psnr_list), 
            np.std(ssim_list)
        ))
    logging.warning('Done.')


if __name__ == "__main__":
    pass
    


