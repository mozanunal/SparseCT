

import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.metrics import (
    mean_squared_error, structural_similarity, peak_signal_noise_ratio)
from skimage.transform import iradon, iradon_sart
import torch
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from pytorch_radon import Radon, IRadon


from .base import Reconstructor
from sparse_ct.loss.tv import tv_3d_l2
from sparse_ct.loss.perceptual import VGGPerceptualLoss
from sparse_ct.tool import im2tensor, plot_result, np_to_torch, torch_to_np
from sparse_ct.model.unet import UNet
from sparse_ct.model.skip import Skip


DEVICE = 'cuda'
DTYPE = torch.cuda.FloatTensor
PAD = 'reflection'
EPOCH = 8000
LR = 0.001
REG_NOISE_STD = 1./100
INPUT_DEPTH = 32
IMAGE_DEPTH = 3
IMAGE_SIZE = 512
N_PROJ = 32
div = 50 #EPOCH / 50

def _FOCUS(img):
    return img[300:450,200:350]

class DipReconstructor(Reconstructor):
    def __init__(self, name, angles):
        super(DipReconstructor, self).__init__(name, angles)

    def calc(self, projs, gt, dip_initial, dip_n_iter=8000, FOCUS=None):      
        # Init Net
        net = Skip(num_input_channels=INPUT_DEPTH,
               num_output_channels=IMAGE_DEPTH,
               upsample_mode='nearest',
               num_channels_down=[16, 32, 64, 128, 256], 
               num_channels_up=[16, 32, 64, 128, 256]).to(DEVICE)
        # net = UNet(num_input_channels=INPUT_DEPTH, num_output_channels=IMAGE_DEPTH,
        #            feature_scale=4, more_layers=0, concat_x=False,
        #            upsample_mode='bilinear', norm_layer=torch.nn.BatchNorm2d,
        #            pad='reflect',
        #            need_sigmoid=False, need_bias=True).to(DEVICE)

        noisy_tensor = np_to_torch(dip_initial).type(DTYPE)
        img_gt_torch = np_to_torch(gt).type(DTYPE)
        net_input = torch.rand(1, INPUT_DEPTH, IMAGE_SIZE, IMAGE_SIZE).type(DTYPE)
        net_input_saved = net_input.detach().clone()
        noise = net_input.detach().clone()

        # Compute number of parameters
        s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
        print ('Number of params: %d' % s)

        # Loss
        mse = torch.nn.MSELoss().to(DEVICE)
        ssim = MS_SSIM(data_range=1.0, size_average=True, channel=IMAGE_DEPTH).to(DEVICE)
        perceptual = VGGPerceptualLoss(resize=True).to(DEVICE)
        theta = torch.linspace(0., 180., N_PROJ).to(DEVICE)
        r = Radon(IMAGE_SIZE, theta, True).to(DEVICE)
        ir = IRadon(IMAGE_SIZE, theta, True).to(DEVICE)
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
        psnr_noisy_hist = []


        best_network = None

        for i in range(dip_n_iter):
            
            # iter
            optimizer.zero_grad()

            if REG_NOISE_STD > 0:
                net_input = net_input_saved + (noise.normal_() * REG_NOISE_STD)

            x_iter = net(net_input)

            if i < 400:
                percep_l = perceptual(x_iter, noisy_tensor.detach())
                proj_l = mse(norm(r(x_iter)[0]), norm(projs[0]))
                loss = proj_l + percep_l
            else:
                proj_l = mse(norm(r(x_iter)[0]), norm(projs[0]))
                loss = proj_l
            # ssim_l = (1 - ssim(x_iter, noisy_tensor.detach() ))
            # tv_l = tv_3d_l2(x_iter[0])
            loss.backward()
            
            optimizer.step()

            # metric
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
                psnr_noisy_hist.append(
                    peak_signal_noise_ratio(x_iter_npy, dip_initial)
                )
                loss_hist.append(loss.item())
                print('{}- psnr: {:.3f} - psnr_noisy: {:.3f} - ssim: {:.3f} - rmse: {:.5f} - loss: {:.5f} '.format(
                    i, psnr_hist[-1], psnr_noisy_hist[-1], ssim_hist[-1], rmse_hist[-1], loss_hist[-1]
                ))

                if psnr_noisy_hist[-1] / max(psnr_noisy_hist) < 0.92:
                    print('Falling back to previous checkpoint.')
                    # load network
                    for new_param, net_param in zip(best_network, net.parameters()):
                        net_param.data.copy_(new_param.cuda())

                if i > 2:
                    if loss_hist[-1] < min(loss_hist[0:-1]):
                        # save network
                        best_network = [x.detach().cpu() for x in net.parameters()]        
                plot_result(gt, dip_initial, x_iter_npy, FOCUS, save_name= save_name+'/{}.png'.format(i))
