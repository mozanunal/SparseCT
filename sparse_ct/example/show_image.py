
import random
import glob
import matplotlib.pyplot as plt
import numpy as np
from skimage.io import imsave
from skimage.transform import radon, iradon
from sparse_ct.data import image_to_sparse_sinogram, ellipses_to_sparse_sinogram
from sparse_ct.reconstructor_2d import IRadonReconstructor


pwd_train = '/external/CT_30_000/train'
pwd_test = '/external/CT_30_000/test'

file_list_train = glob.glob(pwd_train+'/*/*/*.png')
file_list_test = glob.glob(pwd_test+'/*/*/*.png')

# fname = '/external/CT_30_000/train/Images_png_02/Images_png/000089_08_01/038.png'
# '/external/CT_30_000/test/Images_png_56/004457_01_01/027.png'

gt, sinogram, theta, FOCUS = ellipses_to_sparse_sinogram('train', channel=1,
        n_proj=100, size=512, angle1=0.0, angle2=180.0, noise_pow=33.0 )

# for i in range(100):
# fname = random.choice(file_list_train)
# gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
#         n_proj=128, size=512, angle1=0.0, angle2=180.0, noise_pow=12.0 )

plt.figure()
plt.imshow(sinogram, cmap='gray')#, vmin=0.0, vmax=1.0)
plt.figure()
plt.imshow(gt, cmap='gray', vmin=0.0, vmax=1.0)
plt.figure()
plt.imshow(iradon(sinogram, theta=theta), cmap='gray')
plt.show()

recon_fbp = IRadonReconstructor('FBP')
recon_fbp.calc(sinogram, theta)
print(recon_fbp.eval(gt))

#     imsave('img/{}.png'.format(i), gt )

