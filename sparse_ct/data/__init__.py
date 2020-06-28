
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, resize
from skimage.color import gray2rgb
from skimage import io
from scipy.io import loadmat


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


def sparse_shepp_logan(
    noise_level=0.35,
    gray=True,
    n_proj=32,
    angle1=0.0,
    angle2=180.0
):
    gt = resize(shepp_logan_phantom(), (512, 512))
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=True)
    noisy = iradon(sinogram, theta=theta)

    def FOCUS(x):
        return x[350:450, 200:300]

    return gt, noisy, FOCUS

def sparse_breast_phantom(
    noise_level=0.35,
    gray=True,
    n_proj=64,
    angle1=0.0,
    angle2=180.0
):
    gt = loadmat('data/bp-160u-dense.mat')['data'][300]
    gt = resize(gt, (316,316)).astype('float')
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=False)
    noisy = iradon(sinogram, theta=theta)
    noisy = resize(noisy, (316,316)).astype('float')
    print(noisy.shape, gt.shape)
    def FOCUS(x):
        return x[175:275, 75:175]

    return gt, noisy, FOCUS
