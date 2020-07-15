
from skimage.transform import iradon, iradon_sart

from .base import Reconstructor



class IRadonReconstructor(Reconstructor):
    def __init__(self, name, angles):
        super(IRadonReconstructor, self).__init__(name, angles)

    def calc(self, projs):
        return iradon(projs, theta=self.angles)

