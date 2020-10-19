
from sparse_ct.data import image_to_sparse_sinogram, ellipses_to_sparse_sinogram
import matplotlib.pyplot as plt
import numpy as np

fname = '/external/CT_30_000/test/Images_png_56/004457_01_01/027.png'

# gt, sinogram, theta, FOCUS = ellipses_to_sparse_sinogram('train', channel=1,
#         n_proj=128, size=512, angle1=0.0, angle2=180.0, noise_pow=25.0 )

gt, sinogram, theta, FOCUS = image_to_sparse_sinogram(fname, channel=1,
        n_proj=128, size=512, angle1=0.0, angle2=180.0, noise_pow=25.0 )

plt.imshow(gt, cmap='gray', vmin=0.0, vmax=1.0)
plt.show()