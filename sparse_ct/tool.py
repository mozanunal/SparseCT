import numpy as np
import torch
import matplotlib.pyplot as plt


def im2tensor(x, grad=False):
    t = torch.from_numpy(x).float()  # .to(device)
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
