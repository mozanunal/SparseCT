
import numpy as np
from skimage.metrics import (
    mean_squared_error,
    structural_similarity,
    peak_signal_noise_ratio)


class BaseDenoiser(object):
    def __init__(self, name):
        self.name = name
        self.image_d = None

    def eval(self, gt):
        assert self.image_d is not None
        return (
            mean_squared_error(gt,self.image_d),
            peak_signal_noise_ratio(gt,self.image_d),
            structural_similarity(gt,self.image_d)
        )
        
    def calc(self):
        pass
        # should be implemented in


