

import numpy as np
from skimage.metrics import (
    mean_squared_error,
    structural_similarity,
    peak_signal_noise_ratio)

class Reconstructor(object):
    def __init__(self, name, angles):
        self.name = name
        self.angles = angles
        self.image_r = None

    def eval(self, gt):
        assert self.image_r
        return (
            mean_squared_error(self.image_r, gt),
            peak_signal_noise_ratio(self.image_r, gt),
            structural_similarity(self.image_r, gt)
        )

    def calc(self):
        pass
        # should be implemented in





