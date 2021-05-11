
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, resize
from skimage.color import gray2rgb
from skimage import io
from scipy.io import loadmat

from dival.datasets.ellipses_dataset import EllipsesDataset
import sys

def db2ratio(db):
    """
    For power:
    db=10*log(ratio)
    ratio=10**(db/10)
    """
    return 10.0**(db/10.0)

def calc_power(x):
    return np.mean(x**2)# (x.mean()**2) + (x.std()**2)

def awgn(x, desired_noise_pow):
    """
    power of noise:
    sig_pow = mean(X)**2 + std(X)**2
    noise_pow = std(Noise)**2
    snr_ratio = sig_pow/noise_pow
    k = (mean(X)**2 + std(X)**2) / std(Noise)**2
    std(Noise)**2 = (mean(X)**2 + std(X)**2) / k 
    """
    # generate noise
    k = db2ratio(desired_noise_pow)
    noise_var = calc_power(x) / k
    noise = np.random.normal(0.0, np.sqrt(noise_var), x.shape )

    # stats
    signal_power = np.log10( calc_power(x) )*10
    noise_power = np.log10( calc_power(noise) )*10
    # print("Signal -> mean: ", x.mean(),  " std: ", x.std())
    # print("Noise  -> mean: ", noise.mean(), " std: ", noise.std() )
    # print("S - N: ", signal_power, noise_power, " snr: ", desired_noise_pow, " k: ", k,  ) 
    return x + noise

def poisson_noise(x, noise_pow):
    pass

def pad_to_square(img, size_big=None):
    if size_big == None:
        size_big = max(img.shape)+0
    h, w = img.shape
    delta_h = size_big - h
    delta_w = size_big - w
    if delta_h != 0:
        h, w = img.shape
        img = np.vstack([
            np.zeros((delta_h//2, w)),
            img,
            np.zeros((delta_h//2, w))
        ])
    if delta_w != 0:
        h, w = img.shape
        img = np.hstack([
            np.zeros((h, delta_w//2)),
            img,
            np.zeros((h, delta_w//2))
        ])
    return img

def create_circular_mask(h, w, center=None, radius=None):
    if center is None: # use the middle of the image
        center = (int(w/2), int(h/2))
    if radius is None: # use the smallest distance between the center and image walls
        radius = min(center[0], center[1], w-center[0], h-center[1])

    Y, X = np.ogrid[:h, :w]
    dist_from_center = np.sqrt((X - center[0])**2 + (Y-center[1])**2)

    mask = dist_from_center <= radius
    return mask


def noisy_zebra(
    noise_level=0.35,
    gray=True,
    channel=1,
):
    gt = io.imread('data/zebra.jpg', as_gray=gray)
    noisy = gt + (np.random.rand(*gt.shape)-0.5) * noise_level

    def FOCUS(x):
        return x[350:450, 200:300]

    return gt, noisy, FOCUS


def noisy_shepp_logan(
    noise_level=0.35,
    gray=True,
    channel=1
):
    gt = resize(shepp_logan_phantom(), (512, 512))
    noisy = gt + (np.random.rand(*gt.shape)-0.5) * noise_level

    def FOCUS(x):
        return x[350:450, 200:300]

    return gt, noisy, FOCUS



elipData = EllipsesDataset(
        image_size = 512,
        train_len = 32000,
        validation_len = 1000,
        test_len = 0,
        )

def ellipses_to_sparse_sinogram(
    part='train',
    gray=True,
    n_proj=128,
    angle1=0.0,
    angle2=180.0,
    channel=1,
    size=512,
    noise_pow=25.0,
    noise_type='gaussian'
):
    mask = create_circular_mask(size,size)
    gt = np.array(next(elipData.generator(part=part))).astype('float64')
    gt = pad_to_square(gt, size_big=size) * mask
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=True)
    sinogram = awgn(sinogram, noise_pow)
    def FOCUS(x):
        return x[200:300, 200:300], (200, 200, 300, 300)

    if channel == 3:
        gt = gray2rgb(gt)

    return gt, sinogram, theta, FOCUS


def image_to_sparse_sinogram(
    image_path,
    gray=True,
    n_proj=128,
    angle1=0.0,
    angle2=180.0,
    channel=1,
    size=512,
    noise_pow=25.0,
    noise_type='gaussian'
):
    mask = create_circular_mask(size,size)
    raw_img = io.imread(image_path, as_gray=gray).astype('float64')
    if raw_img.max() > 300: # for low dose ct dataset
        pass
        raw_img = raw_img -32768.0 # convert HU
        uplim = 600
        downlim = 300
        raw_img = (raw_img + downlim) / (uplim+downlim)
        raw_img = np.clip(raw_img, 0.0, 1.0)
    else:
        raw_img = raw_img / raw_img.max()
    gt = resize(pad_to_square(raw_img), (size, size)) * mask
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=True)
    if noise_pow:
        sinogram = awgn(sinogram, noise_pow)
    def FOCUS(x):
        return x[200:300, 200:300], (200, 200, 300, 300)

    if channel == 3:
        gt = gray2rgb(gt)

    return gt, sinogram, theta, FOCUS


if __name__ == "__main__":
    pass
