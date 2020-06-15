
import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage import io
from skimage.metrics import (
    mean_squared_error, structural_similarity, peak_signal_noise_ratio)
from tool import im2tensor
from loss.tv import tv_2d_l2
from loss.perceptual import VGGPerceptualLoss

DEVICE = 'cuda'
TV_LAMBDA = 0.2
EPOCH = 3000
div = EPOCH / 20

if __name__ == "__main__":
    im = io.imread('data/zebra.jpg', as_gray=True)
    plt.figure(figsize=(10, 10))
    plt.imshow(im, cmap='gray')
    # Init
    x_real = im2tensor(im).to(DEVICE)
    x_initial = x_real.detach().clone() + 0.25 * torch.randn_like(x_real)
    x_iter = x_initial.detach().clone()
    x_iter.requires_grad_(True)
    x_initial.requires_grad_(False)
    x_real.requires_grad_(False)

    # Metric
    loss_hist = []
    rmse_hist = []
    ssim_hist = []
    psnr_hist = []
    optimizer = torch.optim.SGD([x_iter], lr=0.01,)

    # Optimize
    l2_norm = torch.nn.MSELoss()
    perceptual_dis = VGGPerceptualLoss().to('cuda')

    for i in range(EPOCH):
        # iter
        optimizer.zero_grad()
        lossD = l2_norm(x_iter, x_initial)
        lossT = TV_LAMBDA * tv_2d_l2(x_iter)
        lossP = perceptual_dis(x_iter.unsqueeze(0), x_initial.unsqueeze(0))
        loss = lossT + lossP
        loss.backward(retain_graph=False)
        optimizer.step()
        if i % div == 0:
            xIterN = np.clip(x_iter.detach().cpu().numpy(), 0, 1)
            xRealN = np.clip(x_real.cpu().numpy(), 0, 1)

            rmse_hist.append(
                mean_squared_error(xIterN, xRealN))
            ssim_hist.append(
                structural_similarity(xIterN, xRealN)
            )
            psnr_hist.append(
                peak_signal_noise_ratio(xIterN, xRealN)
            )

            loss_hist.append(loss.item())
            print('{}- psnr: {:.3f} - ssim: {:.3f} - rmse: {:.3f} - loss: {:.3f} '.format(
                i, psnr_hist[-1], ssim_hist[-1], rmse_hist[-1], loss_hist[-1]
            ))
            plt.figure(figsize=(20, 5))
            ims = np.hstack(
                [x_real.cpu().numpy(), x_iter.cpu().detach().numpy(),
                 x_initial.cpu().numpy()])
            plt.imshow(np.clip(ims, 0, 1), cmap='gray')
            plt.show()
    # Result
    plt.figure()
    plt.plot(loss_hist)
    plt.figure()
    plt.plot(rmse_hist)

    plt.figure(figsize=(20, 5))
    ims = np.hstack(
        [x_real.cpu().numpy(), x_iter.cpu().detach().numpy(), x_initial.cpu().numpy()])
    plt.imshow(np.clip(ims, 0, 1), cmap='gray')
    plt.show()
