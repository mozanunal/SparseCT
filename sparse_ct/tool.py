import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def np_to_torch(img_np):
    '''
    '''
    return torch.from_numpy(img_np).permute(2,0,1)[None, :]

def torch_to_np(img_var):
    '''
    '''
    return img_var.detach()[0].cpu().permute(1,2,0).numpy()

def im2tensor(x, grad=False):
    t = torch.from_numpy(x).float()
    t.requires_grad = grad
    return t


def plot_result(gt, noisy, result, FOCUS=None, show=False, save_name=None):
    fig, ax = plt.subplots(2, 1)
    ims = np.hstack([gt, noisy, result])
    focussed_ims = np.hstack(
        [FOCUS(gt), FOCUS(noisy), FOCUS(result)])
    ax[0].imshow(np.clip(ims, 0, 1), cmap='gray')
    ax[1].imshow(np.clip(focussed_ims, 0, 1), cmap='gray')
    if show:
        plt.show()
    if save_name:
        plt.savefig(save_name)
