
import random
import glob
from skimage.io import imsave
from sparse_ct.data import image_to_sparse_sinogram, ellipses_to_sparse_sinogram
import matplotlib.pyplot as plt
import numpy as np

pwd_train = '/external/CT_30_000/train'
pwd_test = '/external/CT_30_000/test'

file_list_train = glob.glob(pwd_train+'/*/*/*.png')
file_list_test = glob.glob(pwd_test+'/*/*/*.png')

# fname = '/external/CT_30_000/train/Images_png_02/Images_png/000089_08_01/038.png'
# '/external/CT_30_000/test/Images_png_56/004457_01_01/027.png'

# gt, sinogram, theta, FOCUS = ellipses_to_sparse_sinogram('train', channel=1,
#         n_proj=128, size=512, angle1=0.0, angle2=180.0, noise_pow=25.0 )

# for i in range(100):
fname = random.choice(file_list_train)
gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
        n_proj=128, size=512, angle1=0.0, angle2=180.0, noise_pow=12.0 )

plt.figure()
plt.imshow(sinogram, cmap='gray')#, vmin=0.0, vmax=1.0)
plt.figure()
plt.hist(gt)
plt.show()

#     imsave('img/{}.png'.format(i), gt )

