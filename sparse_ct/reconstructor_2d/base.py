

import numpy as np
from skimage.metrics import (
    mean_squared_error,
    structural_similarity,
    peak_signal_noise_ratio)

class Reconstructor(object):
    def __init__(self, name):
        self.name = name
        self.image_r = None

    def eval(self, gt):
        assert self.image_r is not None
        return (
            mean_squared_error(gt,self.image_r),
            peak_signal_noise_ratio(gt,self.image_r),
            structural_similarity(gt,self.image_r)
        )

    def evalV2(self, gt, focus):
        assert self.image_r is not None
        img_r_focussed, _ = focus(self.image_r)
        gt_focussed, _ = focus(gt)
        return (
            (
                mean_squared_error(gt_focussed, img_r_focussed),
                peak_signal_noise_ratio(gt_focussed, img_r_focussed),
                structural_similarity(gt_focussed, img_r_focussed)
            ),
            (
                mean_squared_error(gt, self.image_r),
                peak_signal_noise_ratio(gt, self.image_r),
                structural_similarity(gt, self.image_r)
            )
        )
    
    def save_result(self):
        assert self.image_r is not None
        np.save(
            "{}.npy".format(self.name),
            self.image_r
            )

    def calc(self, sinogra, angles):
        pass
        # should be implemented in





