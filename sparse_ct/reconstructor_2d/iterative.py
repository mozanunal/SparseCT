

from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import iradon, iradon_sart
from skimage.restoration import denoise_tv_bregman
from bm3d import bm3d

from sparse_ct.loss.tv import tv_2d_l2
from .base import Reconstructor


class SartReconstructor(Reconstructor):
    def __init__(self, name,
                 sart_n_iter=10, sart_relaxation=0.15):
        super(SartReconstructor, self).__init__(name)
        self.n_iter = sart_n_iter
        self.relaxation = sart_relaxation
        self.hist = {
            "mse": [],
            "psnr": [],
            "ssim": [],
        }

    def calc(self, projs, theta, sart_plot=False):
        image_r = iradon(projs, theta=theta)
        image_r = None
        print('Reconstructing...', self.name)
        for _ in tqdm(range(self.n_iter)):
            image_r = iradon_sart(projs, theta=theta, image=image_r, relaxation=self.relaxation)
            if sart_plot:
                plt.figure()
                plt.imshow(image_r, cmap='gray')
        self.image_r = image_r
        return self.image_r


class SartTVReconstructor(SartReconstructor):
    def __init__(self, name, 
                sart_n_iter=10, sart_relaxation=0.15,
                tv_weight=0.2, tv_n_iter=100):
        super(SartTVReconstructor, self).__init__(name, sart_n_iter=sart_n_iter, sart_relaxation=sart_relaxation)
        self.tv_weight = tv_weight
        self.tv_n_iter = tv_n_iter

    def calc(self, projs, theta, sart_plot=False):
        image_r = super(SartTVReconstructor, self).calc(projs, theta, sart_plot=sart_plot)
        #denoise with tv
        self.image_r = denoise_tv_bregman(image_r, self.tv_weight, self.tv_n_iter)
        return self.image_r


class SartBM3DReconstructor(SartReconstructor):
    def __init__(self, name, 
                sart_n_iter=10, sart_relaxation=0.15,
                bm3d_sigma=0.1):
        super(SartBM3DReconstructor, self).__init__(name, sart_n_iter=sart_n_iter, sart_relaxation=sart_relaxation)
        self.bm3d_sigma = bm3d_sigma

    def calc(self, projs, theta, sart_plot=False):
        image_r = super(SartBM3DReconstructor, self).calc(projs, theta, sart_plot=sart_plot)
        #denoise with tv
        self.image_r = bm3d(image_r, self.bm3d_sigma)
        return self.image_r