
import numpy as np
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, iradon, iradon_sart, resize
from skimage.color import gray2rgb
from skimage import io
from scipy.io import loadmat


def pad_to_square(img):
    size_big = max(img.shape)+0
    h, w = img.shape
    delta_h = size_big - h
    delta_w = size_big - w
    if delta_h != 0:
        h, w = img.shape
        img = np.vstack([
            np.zeros((delta_h//2,w)),
            img,
            np.zeros((delta_h//2,w))
        ])
    if delta_w != 0:
        h, w = img.shape
        img = np.hstack([
            np.zeros((h,delta_w//2)),
            img,
            np.zeros((h,delta_w//2))
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
    gt = resize(gt, (316,316)).astype('float')
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=True)
    noisy = iradon(sinogram, theta=theta)
    noisy = resize(noisy, (316,316)).astype('float')
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
    raw_img = io.imread(image_path, as_gray=gray)
    gt = resize(pad_to_square(raw_img), (size,size))
    theta = np.linspace(angle1, angle2, n_proj, endpoint=False)
    sinogram = radon(gt, theta=theta, circle=True)
    noisy = iradon(sinogram, theta=theta)

    def FOCUS(x):
        return x[250:350, 250:350]


    if channel == 3:
        noisy = gray2rgb(noisy)
        gt = gray2rgb(gt)

    return gt, noisy, FOCUS

if __name__ == "__main__":
    gt,n,_ = sparse_image("data/zebra.jpg")
    #gt,n,_ = sparse_shepp_logan("data/zebra.jpg")
    import matplotlib.pyplot as plt

    plt.imshow(gt, cmap='gray')
    plt.show()
    plt.imshow(n, cmap='gray')
    plt.show()



