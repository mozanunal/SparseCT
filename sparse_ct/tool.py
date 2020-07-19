import numpy as np
import torch
import matplotlib.pyplot as plt
import torchvision.transforms.functional as F

def np_to_torch(img_np):
    '''
    np                torch
    ---                ---
    512 x 512 x 3 ->  1 x 3 x 512 x 512
    512 x 512     ->  1 x 1 x 512 x 512
    '''
    assert len(img_np.shape) in [2,3]
    if len(img_np.shape) == 3:
        return torch.from_numpy(img_np).permute(2,0,1)[None, :]
    elif len(img_np.shape) == 2:
        return torch.from_numpy(img_np)[None, None, :]
    else:
        assert False
        
def torch_to_np(img_var):
    '''
    torch                 np
    ---                   ---
    1 x 3 x 512 x 512 ->  512 x 512 x 3
    1 x 1 x 512 x 512 ->  512 x 512
    '''
    assert len(img_var.shape) == 4
    assert img_var.shape[1] in [1,3]
    if img_var.shape[1] == 3:
        return img_var.detach()[0].cpu().permute(1,2,0).numpy()
    elif img_var.shape[1] == 1:
        return img_var.detach()[0,0].cpu().numpy()
    else:
        assert False

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
    plt.close()

def plot_grid(imgs, FOCUS=None, show=False, save_name=None, dpi=300):
    fig, ax = plt.subplots(2, 1)
    ims = np.hstack(imgs)
    focussed_ims = np.hstack(
        [FOCUS(x) for x in imgs]
    )
    ax[0].imshow(np.clip(ims, 0, 1), cmap='gray')
    ax[1].imshow(np.clip(focussed_ims, 0, 1), cmap='gray')
    if show:
        plt.show()
    if save_name:
        plt.savefig(save_name, dpi=dpi)
    plt.close()
