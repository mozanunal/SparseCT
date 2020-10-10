
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
    db=10*log(ratio)
    ratio=10**(db/10)
    """
    return 10.0**(db/10.0)

def awgn(x, noise_pow):
    try:
        k = db2ratio(noise_pow)
        return x + np.random.normal(0.0, x.mean()/k, x.shape )
    except Exception as e:
        print('awgn error', e, file=sys.stderr,)
        return x
 

def pad_to_square(img):
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


def sparse_shepp_logan(
    noise_level=0.35,
    gray=True,
    n_proj=32,
    angle1=0.0,
    angle2=180.0,
    channel=1,
    size=512
):
    gt = resize(shepp_logan_phantom(), (size, size))
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=True)
    noisy = iradon(sinogram, theta=theta)

    def FOCUS(x):
        return x[350:450, 200:300]

    if channel == 3:
        noisy = gray2rgb(noisy)
        gt = gray2rgb(gt)

    return gt, noisy, FOCUS


def sparse_breast_phantom(
    noise_level=0.35,
    gray=True,
    n_proj=64,
    angle1=0.0,
    angle2=180.0,
    channel=1,
):
    gt = loadmat('data/bp-160u-dense.mat')['data'][300]
    gt = resize(gt, (316, 316)).astype('float')
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=True)
    noisy = iradon(sinogram, theta=theta)
    noisy = resize(noisy, (316, 316)).astype('float')
    print(noisy.shape, gt.shape)

    def FOCUS(x):
        return x[175:275, 75:175]

    return gt, noisy, FOCUS


def sparse_image(
    image_path,
    noise_level=0.25,
    gray=True,
    n_proj=128,
    angle1=0.0,
    angle2=180.0,
    channel=1,
    size=512
):
    raw_img = io.imread(image_path, as_gray=gray).astype('float64')
    if raw_img.max() > 300: # for low dose ct dataset
        raw_img = raw_img - 31744.0
        raw_img = raw_img / 4096.0
    else:
        raw_img = raw_img / raw_img.max()
    gt = resize(pad_to_square(raw_img), (size, size))
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=True)
    noisy = None#iradon(sinogram, theta=theta)
    for _ in range(40):
        noisy = iradon_sart(sinogram, theta=theta, image=noisy, relaxation=0.03)

    def FOCUS(x):
        return x[250:350, 250:350]

    if channel == 3:
        noisy = gray2rgb(noisy)
        gt = gray2rgb(gt)

    return gt, noisy, FOCUS




elipData = EllipsesDataset(
        image_size = 512,
        train_len = 32000,
        validation_len = 3200,
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
    noise_pow=15.0
):
    gt = np.array(next(elipData.generator(part=part))).astype('float64')
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
    noise_pow=15.0
):
    raw_img = io.imread(image_path, as_gray=gray).astype('float64')
    if raw_img.max() > 300: # for low dose ct dataset
        raw_img = raw_img - 31744.0
        raw_img = raw_img / 4096.0
    else:
        raw_img = raw_img / raw_img.max()
    gt = resize(pad_to_square(raw_img), (size, size))
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=True)
    sinogram = awgn(sinogram, noise_pow)
    def FOCUS(x):
        return x[200:300, 200:300], (200, 200, 300, 300)

    if channel == 3:
        gt = gray2rgb(gt)

    return gt, sinogram, theta, FOCUS


if __name__ == "__main__":
    gt, n, _ = sparse_image("data/pomegranate.jpg")
    #gt,n,_ = sparse_shepp_logan("data/zebra.jpg")
    import matplotlib.pyplot as plt

    plt.imshow(gt, cmap='gray')
    plt.show()
    plt.imshow(n, cmap='gray')
    plt.show()
