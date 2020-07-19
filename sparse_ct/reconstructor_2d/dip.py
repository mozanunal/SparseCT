

import matplotlib.pyplot as plt
import numpy as np
import os
from skimage.color import rgb2gray, gray2rgb
from skimage.metrics import (
    mean_squared_error, structural_similarity, peak_signal_noise_ratio)
from skimage.transform import iradon, iradon_sart
import torch
from torchvision import transforms
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from pytorch_radon import Radon, IRadon

from tqdm import tqdm
from .base import Reconstructor
from sparse_ct.loss.tv import tv_2d_l2
from sparse_ct.loss.perceptual import VGGPerceptualLoss
from sparse_ct.tool import im2tensor, plot_result, np_to_torch, torch_to_np
from sparse_ct.model.unet import UNet
from sparse_ct.model.skip import Skip


DEVICE = 'cuda'
DTYPE = torch.cuda.FloatTensor
EPOCH = 8000
INPUT_DEPTH = 32
IMAGE_DEPTH = 1
IMAGE_SIZE = 512
N_PROJ = 32
ANGLE1 = 0.
ANGLE2 = 180.
div = 50 #EPOCH / 50

def _FOCUS(img):
    return img[300:450,200:350]

class DipReconstructor(Reconstructor):
    def __init__(self, name, angles,
        dip_n_iter=8000, net='skip',
        lr=0.001, reg_std=1./100,
         w_proj_loss=0.3, w_perceptual_loss=0.33, w_tv_loss=0.33):
        super(DipReconstructor, self).__init__(name, angles)
        self.n_iter = dip_n_iter
        assert net in ['skip', 'unet']
        self.net = net
        self.lr = lr
        self.reg_std = reg_std
        self.w_proj_loss = w_proj_loss
        self.w_perceptual_loss = w_perceptual_loss
        self.w_tv_loss = w_tv_loss
        self.gt = None
        self.noisy = None
        self.FOCUS = None
        self.log_dir = None
 
    def set_for_metric(self, gt, dip_initial, 
                      FOCUS=None,
                      log_dir='log/dip'):
        assert len(gt.shape) == 2
        assert len(dip_initial.shape) == 2
        print
        self.gt = gt
        self.noisy = dip_initial
        self.FOCUS = FOCUS
        self.log_dir = log_dir

    def calc(self, projs):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        net = self._get_net()

        noisy_tensor = np_to_torch(self.noisy).type(DTYPE)
        img_gt_torch = np_to_torch(self.gt).type(DTYPE)
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
        theta = torch.linspace(ANGLE1, ANGLE2, N_PROJ).to(DEVICE)
        r = Radon(IMAGE_SIZE, theta, True).to(DEVICE)
        ir = IRadon(IMAGE_SIZE, theta, True).to(DEVICE)
        # Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)
        cur_lr = self.lr
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

        print('Reconstructing with DIP...')
        for i in tqdm(range(self.n_iter)):
            
            # iter
            optimizer.zero_grad()

            if self.reg_std > 0:
                net_input = net_input_saved + (noise.normal_() * self.reg_std)

            x_iter = net(net_input)

            if i < 100:
                percep_l = perceptual(x_iter, noisy_tensor.detach())
                proj_l = mse(norm(r(x_iter)[0]), norm(projs[0]))
                loss = proj_l + percep_l
            else:
                percep_l = 0
                proj_l = 0
                tv_l = 0
                if self.w_perceptual_loss > 0.0:
                    percep_l = perceptual(x_iter, noisy_tensor.detach())
                if self.w_proj_loss > 0.0:
                    proj_l = mse(norm(r(x_iter)[0]), norm(projs[0]))
                if self.w_tv_loss > 0.0:
                    tv_l = tv_2d_l2(x_iter[0].mean(axis=0))
                loss = self.w_proj_loss* proj_l + self.w_perceptual_loss * percep_l + self.w_tv_loss * tv_l
            # ssim_l = (1 - ssim(x_iter, noisy_tensor.detach() ))
            
            loss.backward()
            
            optimizer.step()

            # metric
            if i % div == 0:
                x_iter_npy = np.clip(torch_to_np(x_iter), 0, 1)

                rmse_hist.append(
                    mean_squared_error(x_iter_npy, self.gt))
                ssim_hist.append(
                    structural_similarity(x_iter_npy, self.gt, multichannel=True)
                )
                psnr_hist.append(
                    peak_signal_noise_ratio(x_iter_npy, self.gt)
                )
                psnr_noisy_hist.append(
                    peak_signal_noise_ratio(x_iter_npy, self.noisy)
                )
                loss_hist.append(loss.item())
                print('{}- psnr: {:.3f} - psnr_noisy: {:.3f} - ssim: {:.3f} - rmse: {:.5f} - loss: {:.5f} '.format(
                    i, psnr_hist[-1], psnr_noisy_hist[-1], ssim_hist[-1], rmse_hist[-1], loss_hist[-1]
                ))

                if psnr_noisy_hist[-1] / max(psnr_noisy_hist) < 0.92:
                    print('Falling back to previous checkpoint.')
                    for g in optimizer.param_groups:
                        g['lr'] = cur_lr / 10.0
                    cur_lr = cur_lr / 10.0
                    print("optimizer.lr", cur_lr)
                    # load network
                    for new_param, net_param in zip(best_network, net.parameters()):
                        net_param.data.copy_(new_param.cuda())

                if i > 2:
                    if loss_hist[-1] < min(loss_hist[0:-1]):
                        # save network
                        best_network = [x.detach().cpu() for x in net.parameters()]        
                plot_result(self.gt, self.noisy, x_iter_npy, self.FOCUS, save_name=self.log_dir+'/{}.png'.format(i))

        self.image_r = x_iter_npy
        return self.image_r


    def _get_net(self):
        # Init Net
        if self.net == 'skip':
            return Skip(num_input_channels=INPUT_DEPTH,
                num_output_channels=IMAGE_DEPTH,
                upsample_mode='nearest',
                num_channels_down=[16, 32, 64, 128, 256], 
                num_channels_up=[16, 32, 64, 128, 256]).to(DEVICE)
        elif self.net == 'unet':
            return UNet(num_input_channels=INPUT_DEPTH, num_output_channels=IMAGE_DEPTH,
                    feature_scale=4, more_layers=0, concat_x=False,
                    upsample_mode='bilinear', norm_layer=torch.nn.BatchNorm2d,
                    pad='reflect',
                    need_sigmoid=False, need_bias=True).to(DEVICE)
        else:
            assert False
