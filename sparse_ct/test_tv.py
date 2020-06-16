
import matplotlib.pyplot as plt
import torch
import numpy as np
from skimage.metrics import (
    mean_squared_error, structural_similarity, peak_signal_noise_ratio)

from data import (noisy_zebra, noisy_shepp_logan)
from tool import im2tensor, plot_result
from loss.tv import tv_2d_l2
from loss.perceptual import VGGPerceptualLoss

DEVICE = 'cuda'
TV_LAMBDA = 0.8
EPOCH = 3000
div = EPOCH / 20

if __name__ == "__main__":
    gt, noisy, FOCUS = noisy_shepp_logan()
    plt.figure(figsize=(10, 10))
    plt.imshow(np.hstack((gt, noisy)), cmap='gray')
    plt.imshow(np.hstack((FOCUS(gt), FOCUS(noisy))), cmap='gray')
    # Init
    x_real = im2tensor(gt).to(DEVICE)
    x_initial = im2tensor(noisy).to(DEVICE)
    x_iter = x_initial.detach().clone()
    x_iter.requires_grad_(True)
    x_initial.requires_grad_(False)
    x_real.requires_grad_(False)

    # Metric
    loss_hist = []
    rmse_hist = []
    ssim_hist = []
    psnr_hist = []
    optimizer = torch.optim.SGD([x_iter], lr=0.001,)

    # Optimize
    l2_norm = torch.nn.MSELoss()
    perceptual_dis = VGGPerceptualLoss().to('cuda')

    for i in range(EPOCH):
        # iter
        optimizer.zero_grad()
        lossD = l2_norm(x_iter, x_initial)
        lossT = TV_LAMBDA * tv_2d_l2(x_iter)
        #lossP = perceptual_dis(x_iter.unsqueeze(0), x_initial.unsqueeze(0))
        loss = lossT + lossD
        loss.backward(retain_graph=False)
        optimizer.step()
        if i % div == 0:
            x_iter_npy = np.clip(x_iter.detach().cpu().numpy(), 0, 1)
            rmse_hist.append(
                mean_squared_error(x_iter_npy, gt))
            ssim_hist.append(
                structural_similarity(x_iter_npy, gt)
            )
            psnr_hist.append(
                peak_signal_noise_ratio(x_iter_npy, gt)
            )

            loss_hist.append(loss.item())
            print('{}- psnr: {:.3f} - ssim: {:.3f} - rmse: {:.3f} - loss: {:.3f} '.format(
                i, psnr_hist[-1], ssim_hist[-1], rmse_hist[-1], loss_hist[-1]
            ))
            plot_result(gt, noisy, x_iter_npy, FOCUS)
    # Result
    plt.figure()
    plt.plot(loss_hist)
    plt.figure()
    plt.plot(rmse_hist)
    plot_result(gt, noisy, x_iter_npy, FOCUS)
