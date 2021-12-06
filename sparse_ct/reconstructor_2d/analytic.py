
from skimage.transform import iradon, iradon_sart
from bm3d import bm3d
from .base import Reconstructor



class IRadonReconstructor(Reconstructor):
    def __init__(self, name):
        super(IRadonReconstructor, self).__init__(name)

    def calc(self, projs, theta):
        self.image_r = iradon(projs, theta=theta)
        return self.image_r


class FBP_BM3DReconstructor(IRadonReconstructor):
    def __init__(self, name, bm3d_sigma=0.1):
        super(FBP_BM3DReconstructor, self).__init__(name)
        self.bm3d_sigma = bm3d_sigma

    def calc(self, projs, theta, sart_plot=False):
        image_r = super(FBP_BM3DReconstructor, self).calc(projs, theta)
        #denoise with tv
        self.image_r = bm3d(image_r, self.bm3d_sigma)
        return self.image_r