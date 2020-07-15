
from skimage.transform import iradon, iradon_sart

from .base import Reconstructor



class IRadonReconstructor(Reconstructor):
    def __init__(self, name, angles):
        super(IRadonReconstructor, self).__init__(name, angles)

    def calc(self, projs):
        self.image_r = iradon(projs, theta=self.angles)
        return self.image_r

