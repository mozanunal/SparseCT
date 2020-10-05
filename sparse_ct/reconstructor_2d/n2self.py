
import os
import random
import numpy as np
from tqdm import tqdm
import torch
from torchvision import transforms
from skimage.metrics import (
    mean_squared_error, 
    structural_similarity, 
    peak_signal_noise_ratio
)

from pytorch_radon import Radon, IRadon

from sparse_ct.model.unet import UNet
from sparse_ct.model.skip import Skip
from sparse_ct.tool import im2tensor, plot_grid, np_to_torch, torch_to_np
from .base import Reconstructor


def toSubLists(full, ratio):
    idx = set(
        random.choices(
            list(full),k=int(ratio*len(full))
        )
    )
    idxC = list(full-idx)
    idx = list(idx)
    return idx, idxC

def _FOCUS(img):
    return img[300:450,200:350]

class N2SelfReconstructor(Reconstructor):
    DEVICE = 'cuda'
    DTYPE = torch.cuda.FloatTensor
    INPUT_DEPTH = 1
    IMAGE_DEPTH = 1
    IMAGE_SIZE = 512
    SHOW_EVERY = 50

    def __init__(self, name, angles,
        n2self_n_iter=8000, net='skip',
        lr=0.001 ):
        super(N2SelfReconstructor, self).__init__(name, angles)
        self.n_proj = len(angles)
        self.n_iter = n2self_n_iter
        assert net in ['skip', 'unet']
        self.net = net
        self.lr = lr
        # loss functions
        self.mse = torch.nn.MSELoss().to(self.DEVICE)
        self.theta = torch.from_numpy(angles).to(self.DEVICE)
        self.gt = None
        self.noisy = None
        self.FOCUS = None
        self.log_dir = None
 
    def set_for_metric(self, gt, noisy, 
                      FOCUS=None,
                      log_dir='log/dip'):
        assert len(gt.shape) == 2
        assert len(noisy.shape) == 2
        self.gt = gt
        self.noisy = noisy
        self.FOCUS = FOCUS
        self.log_dir = log_dir

    def calc(self, projs):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)

        net = self._get_net()

        projs_torch = np_to_torch(projs).type(self.DTYPE)
        norm = transforms.Normalize(projs_torch[0].mean((1,2)), projs_torch[0].std((1,2)))
        full = set(i for i in range(self.n_proj))


        # Compute number of parameters
        s  = sum([np.prod(list(p.size())) for p in net.parameters()]); 
        print ('Number of params: %d' % s)

        # Optimizer
        optimizer = torch.optim.Adam(net.parameters(), lr=self.lr)

        # Iterations
        loss_hist = []
        rmse_hist = []
        ssim_hist = []
        psnr_hist = []
        psnr_noisy_hist = []
        best_result = None

        print('Reconstructing with DIP...')
        for i in tqdm(range(self.n_iter)):
            
            # iter
            optimizer.zero_grad()

            # train
            if i % self.SHOW_EVERY != 0:
                idx, idxC = toSubLists(full, 0.1)
                tsub = self.theta[idx]
                tsubC = self.theta[idxC]
            # val
            else:
                idx, idxC = list(full), list(full)
                tsub = self.theta
                tsubC = self.theta

            r = Radon(self.IMAGE_SIZE, tsub, True).to(self.DEVICE)
            ir = IRadon(self.IMAGE_SIZE, tsubC, True).to(self.DEVICE)

            x_iter = net(
                ir(projs_torch[:,:,:,idxC])
            )
            loss = self.mse(
                norm(r(x_iter)[0]).unsqueeze(0),
                norm(projs_torch[:,:,:,idx][0]).unsqueeze(0)
            )
            
            loss.backward()
            
            optimizer.step()

            # metric
            if i % self.SHOW_EVERY == 0:
                x_iter_npy = np.clip(torch_to_np(x_iter), 0, 1).astype(np.float64)
                print(x_iter_npy.shape)
                rmse_hist.append(
                    mean_squared_error(x_iter_npy, self.gt))
                ssim_hist.append(
                    structural_similarity(x_iter_npy, self.gt, multichannel=True)
                )
                psnr_hist.append(
                    peak_signal_noise_ratio(x_iter_npy, self.gt)
                )
                loss_hist.append(loss.item())
                print('{}/{}- psnr: {:.3f} - ssim: {:.3f} - rmse: {:.5f} - loss: {:.5f} '.format(
                    self.name, i, psnr_hist[-1], ssim_hist[-1], rmse_hist[-1], loss_hist[-1]
                ))

                if i > 2:
                    if loss_hist[-1] < min(loss_hist[0:-1]):
                        # save network
                        # best_network = [x.detach().cpu() for x in net.parameters()]
                        best_result = x_iter_npy.copy() 
                plot_grid([self.gt, x_iter_npy], self.FOCUS, save_name=self.log_dir+'/{}.png'.format(i))

        self.image_r = best_result
        return self.image_r

    def _get_net(self):
        # Init Net
        if self.net == 'skip':
            return Skip(num_input_channels=self.INPUT_DEPTH,
                num_output_channels=self.IMAGE_DEPTH,
                upsample_mode='nearest',
                num_channels_down=[16, 32, 64, 128, 256], 
                num_channels_up=[16, 32, 64, 128, 256]).to(self.DEVICE)
        elif self.net == 'unet':
            return UNet(num_input_channels=self.INPUT_DEPTH, num_output_channels=self.IMAGE_DEPTH,
                    feature_scale=4, more_layers=0, concat_x=False,
                    upsample_mode='bilinear', norm_layer=torch.nn.BatchNorm2d,
                    pad='reflect',
                    need_sigmoid=False, need_bias=True).to(self.DEVICE)
        else:
            assert False
