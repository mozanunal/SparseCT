
from skimage.transform import iradon, iradon_sart

from .base import Reconstructor



class IRadonReconstructor(Reconstructor):
    def __init__(self, name):
        super(IRadonReconstructor, self).__init__(name)

    def calc(self, projs, theta):
        self.image_r = iradon(projs, theta=theta)
        return self.image_r

