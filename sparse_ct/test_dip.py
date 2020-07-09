
import matplotlib.pyplot as plt
import numpy as np
from skimage.metrics import (
    mean_squared_error, structural_similarity, peak_signal_noise_ratio)
import torch
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from pytorch_radon import Radon, IRadon

from data import (noisy_zebra, noisy_shepp_logan, sparse_shepp_logan, sparse_breast_phantom)
from tool import im2tensor, plot_result, np_to_torch, torch_to_np
from model.unet import UNet

INPUT_DEPTH = 32
IMAGE_DEPTH = 3
DEVICE = 'cuda'
DTYPE = torch.cuda.FloatTensor
PAD = 'reflection'
EPOCH = 6000
LR = 0.001
CHANNEL = 3
reg_noise_std = 1./50

div = EPOCH / 20

if __name__ == "__main__":

    # Init Input 
    gt, noisy, FOCUS = sparse_shepp_logan(channel=CHANNEL)
    
    plt.figure(figsize=(10, 10))
    plt.imshow(np.hstack((gt, noisy)), cmap='gray')
    plt.show()
    plt.imshow(np.hstack((FOCUS(gt), FOCUS(noisy))), cmap='gray')
    plt.show()
    

    # Init Net
    net = UNet(num_input_channels=INPUT_DEPTH, num_output_channels=CHANNEL,
               feature_scale=4, more_layers=0, concat_x=False,
               upsample_mode='bilinear', norm_layer=torch.nn.BatchNorm2d,
               pad='reflect',
               need_sigmoid=False, need_bias=True).to(DEVICE)

    noisy_tensor = np_to_torch(noisy).type(DTYPE)
    img_gt_torch = np_to_torch(gt).type(DTYPE)
    net_input = torch.rand(IMAGE_DEPTH, INPUT_DEPTH, 512, 512).type(DTYPE)
    net_input_saved = net_input.detach().clone()
    noise = net_input.detach().clone()

    # Compute number of parameters
    s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
    print ('Number of params: %d' % s)

    # Loss
    mse = torch.nn.MSELoss().to(DEVICE)
    #ssim = MS_SSIM(data_range=1.0, size_average=True, channel=IMAGE_DEPTH).to(DEVICE)
    theta = torch.linspace(0., 180., 32)
    r = Radon(512, theta, True)
    ir = IRadon(512, theta, True)
    # Optimizer
    optimizer = torch.optim.Adam(net.parameters(), lr=LR)

    projs = r(img_gt_torch).detach().clone()
    norm = transforms.Normalize(projs[0].mean((1,2)), projs[0].std((1,2)))
    print(noisy_tensor.shape, net_input.shape, projs.shape)

    # Iterations
    loss_hist = []
    rmse_hist = []
    ssim_hist = []
    psnr_hist = []

    for i in range(EPOCH):
        # iter
        optimizer.zero_grad()

        if reg_noise_std > 0:
            net_input = net_input_saved + (noise.normal_() * reg_noise_std)

        x_iter = net(net_input)
        proj_l = mse(norm(r(x_iter)[0]), norm(projs[0]))
        loss = proj_l
        loss.backward()
        
        optimizer.step()


        if i % div == 0:
            x_iter_npy = np.clip(torch_to_np(x_iter), 0, 1)

            rmse_hist.append(
                mean_squared_error(x_iter_npy, gt))
            ssim_hist.append(
                structural_similarity(x_iter_npy, gt, multichannel=True)
            )
            psnr_hist.append(
                peak_signal_noise_ratio(x_iter_npy, gt)
            )

            loss_hist.append(loss.item())
            print('{}- psnr: {:.4f} - ssim: {:.4f} - rmse: {:.4f} - loss: {:.4f} '.format(
                i, psnr_hist[-1], ssim_hist[-1], rmse_hist[-1], loss_hist[-1]
            ))
    plot_result(gt, noisy, x_iter_npy, FOCUS)
    # Result
    plt.figure()
    plt.plot(loss_hist)
    plt.figure()
    plt.plot(rmse_hist)
    plot_result(gt, noisy, x_iter_npy, FOCUS)