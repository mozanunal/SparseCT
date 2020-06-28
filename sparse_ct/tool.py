import numpy as np
import torch
import matplotlib.pyplot as plt


def np_to_torch(img_np):
    '''Converts image in numpy.array to torch.Tensor.

    From C x W x H [0..1] to  C x W x H [0..1]
    '''
    return torch.from_numpy(img_np)[None, :]

def torch_to_np(img_var):
    '''Converts an image in torch.Tensor format to np.array.

    From 1 x C x W x H [0..1] to  C x W x H [0..1]
    '''
    return img_var.detach().cpu().numpy()[0]


def im2tensor(x, grad=False):
    t = torch.from_numpy(x).float()
    t.requires_grad = grad
    return t


def plot_result(gt, noisy, result, FOCUS=None):
    fig, ax = plt.subplots(2, 1)
    ims = np.hstack([gt, noisy, result])
    focussed_ims = np.hstack(
        [FOCUS(gt), FOCUS(noisy), FOCUS(result)])
    ax[0].imshow(np.clip(ims, 0, 1), cmap='gray')
    ax[1].imshow(np.clip(focussed_ims, 0, 1), cmap='gray')
    plt.show()
