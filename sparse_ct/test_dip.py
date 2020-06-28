
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import (
    mean_squared_error, structural_similarity, peak_signal_noise_ratio)
import torch
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

from data import (noisy_zebra, noisy_shepp_logan, sparse_shepp_logan, sparse_breast_phantom)
from tool import im2tensor, plot_result, np_to_torch
from model.unet import UNet

INPUT_DEPTH = 32
IMAGE_DEPTH = 1
DEVICE = 'cuda'
DTYPE = dtype = torch.cuda.FloatTensor
PAD = 'reflection'
EPOCH = 2000
LR = 0.001

div = EPOCH / 40

if __name__ == "__main__":

    # Init Input 
    gt, noisy, FOCUS = sparse_shepp_logan()
    
    # plt.figure(figsize=(10, 10))
    # plt.imshow(np.hstack((gt, noisy)), cmap='gray')
    # plt.show()
    # plt.imshow(np.hstack((FOCUS(gt), FOCUS(noisy))), cmap='gray')
    # plt.show()
    

    # Init Net
    net = UNet(num_input_channels=INPUT_DEPTH, num_output_channels=1,
               feature_scale=4, more_layers=0, concat_x=False,
               upsample_mode='bilinear', norm_layer=torch.nn.BatchNorm2d,
               pad='reflect',
               need_sigmoid=False, need_bias=True).to(DEVICE)

    noisy_tensor = np_to_torch(noisy).unsqueeze(0).type(DTYPE)
    net_input = torch.rand(IMAGE_DEPTH, INPUT_DEPTH, 512, 512).type(DTYPE)
    print(noisy_tensor.shape, net_input.shape)

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().to(DEVICE)
    ssim = MS_SSIM(data_range=1.0, size_average=True, channel=IMAGE_DEPTH).to(DEVICE)

    # Optimizer
    optimizer = torch.optim.Adam([net_input], lr=LR)

    # Iterations
    loss_hist = []
    rmse_hist = []
    ssim_hist = []
    psnr_hist = []

    for i in range(EPOCH):
        # iter
        optimizer.zero_grad()

        x_iter = net(net_input)
        loss = mse(x_iter, noisy_tensor)
        loss.backward()
        
        optimizer.step()


        if i % div == 0:
            x_iter_npy = np.clip(x_iter.detach().cpu().numpy(), 0, 1)[0,0]
            print(x_iter_npy.dtype, gt.dtype)

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