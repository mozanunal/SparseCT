
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, resize
from skimage.color import gray2rgb
from skimage import io


def noisy_zebra(
    noise_level=0.35,
    gray=True
):
    gt = io.imread('data/zebra.jpg', as_gray=gray)
    noisy = gt + (np.random.rand(*gt.shape)-0.5) * noise_level

    def FOCUS(x):
        return x[350:450, 200:300]

    return gt, noisy, FOCUS


def noisy_shepp_logan(
    noise_level=0.35,
    gray=True
):
    gt = resize(shepp_logan_phantom(), (512, 512))
    noisy = gt + (np.random.rand(*gt.shape)-0.5) * noise_level

    def FOCUS(x):
        return x[350:450, 200:300]

    return gt, noisy, FOCUS
