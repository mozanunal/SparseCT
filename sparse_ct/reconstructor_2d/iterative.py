
from tqdm import tqdm
import matplotlib.pyplot as plt
from skimage.transform import iradon, iradon_sart
from .base import Reconstructor

class SartReconstructor(Reconstructor):
    def __init__(self, name, angles,
                 sart_n_iter=10, sart_relaxation=0.15):
        super(SartReconstructor, self).__init__(name, angles)
        self.n_iter = sart_n_iter
        self.relaxation = sart_relaxation
        self.hist = {
            "mse": [],
            "psnr": [],
            "ssim": [],
        }

    def calc(self, projs, sart_plot=False):
        image_r = iradon(projs, theta=self.angles)
        image_r = None
        print('Reconstructing...', self.name)
        for _ in tqdm(range(self.n_iter)):
            image_r = iradon_sart(projs, theta=self.angles, image=image_r, relaxation=self.relaxation)
            if sart_plot:
                plt.figure()
                plt.imshow(image_r, cmap='gray')
        self.image_r = image_r
        return self.image_r


class SartTVReconstructor(Reconstructor):
    def __init__(self, name, projs, angles, n_iter=10, tv_weight=0.2, n_iter_tv=40):
        pass
