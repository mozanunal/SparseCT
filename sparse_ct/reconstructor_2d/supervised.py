
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
from pytorch_radon.filters import LearnableFilter

from sparse_ct.model.unet import UNet
from sparse_ct.model.skip import Skip
from sparse_ct.tool import im2tensor, plot_grid, np_to_torch, torch_to_np
from .base import Reconstructor


def _FOCUS(img):
    return img[300:450,200:350]


class SupervisedReconstructor(Reconstructor):
    DEVICE = 'cuda'
    DTYPE = torch.cuda.FloatTensor
    INPUT_DEPTH = 1
    IMAGE_DEPTH = 1
    IMAGE_SIZE = 512
    SHOW_EVERY = 50
    SAVE_EVERY = 1000

    def __init__(self, name,
        net='skip', lr=0.001,
        weights=None):
        super(SupervisedReconstructor, self).__init__(name)
        assert net in ['skip', 'unet', 'skipV2']
        self.lr = lr
        
        # net
        self.net_type = net
        self.weights = weights

        # for testing
        self.gt = None
        self.noisy = None
        self.FOCUS = None
        self.log_dir = None

        # Iterations
        self.loss_hist = []
        self.rmse_hist = []
        self.ssim_hist = []
        self.psnr_hist = []
        self.best_result = None
 
    ######### Inference #########
    def set_for_metric(self, gt, noisy, 
                      FOCUS=None,
                      log_dir='log/dip'):
        assert len(gt.shape) == 2
        assert len(noisy.shape) == 2
        self.gt = gt
        self.noisy = noisy
        self.FOCUS = FOCUS
        self.log_dir = log_dir

    def init_train(self, theta):
        self.theta = torch.from_numpy(theta).type(self.DTYPE)
        self.n_proj = len(theta)
        self.loss_hist = []
        self.rmse_hist = []
        self.ssim_hist = []
        self.psnr_hist = []
        self.net = self._get_net(self.net_type)
        self.i_iter = 0
        self.ir = IRadon(self.IMAGE_SIZE, theta, True).to(self.DEVICE)
        self.mse = torch.nn.MSELoss().to(self.DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        if self.weights:
            self._load(self.weights)

    def calc(self, projs, theta):
        
        self.init_train(theta)

        self.net.eval()
        projs = np_to_torch(projs).type(self.DTYPE)
        x_iter = self.net(
            self.ir(projs)
        )
        x_iter_npy = np.clip(torch_to_np(x_iter), 0, 1).astype(np.float64)
        self.image_r = x_iter_npy.copy()
        return self.image_r

    def _get_net(self, net):
        # Init Net
        if net == 'skip':
            return Skip(num_input_channels=self.INPUT_DEPTH,
                num_output_channels=self.IMAGE_DEPTH,
                upsample_mode='nearest',
                num_channels_down=[16, 32, 64, 128, 256], 
                num_channels_up=[16, 32, 64, 128, 256]).to(self.DEVICE)
        elif net == 'skipV2':
            return Skip(num_input_channels=self.INPUT_DEPTH,
                num_output_channels=self.IMAGE_DEPTH,
                upsample_mode='nearest',
                num_channels_down=[32, 64, 128, 256, 512], 
                num_channels_up=[32, 64, 128, 256, 512]).to(self.DEVICE)
        elif net == 'unet':
            return UNet(num_input_channels=self.INPUT_DEPTH, num_output_channels=self.IMAGE_DEPTH,
                    feature_scale=4, more_layers=0, concat_x=False,
                    upsample_mode='bilinear', norm_layer=torch.nn.BatchNorm2d,
                    pad='reflect',
                    need_sigmoid=False, need_bias=True).to(self.DEVICE)
        else:
            assert False



    ######### Training #########
    def _save(self, save_name):
        torch.save(self.net.state_dict(), save_name)

    def _load(self, load_name):
        print('weights are loaded...')
        self.net.load_state_dict(torch.load(load_name))

    def train(self, train_loader, test_loader, epochs=10):
        pass

    def _train_one_epoch(self, train_loader, test_loader):
        self.net.train()

        for projs, gts in tqdm(train_loader):
            self.i_iter += 1

            projs = projs.type(self.DTYPE)
            gts = gts.type(self.DTYPE)

            self.optimizer.zero_grad()            

            x_iter = self.net(
                self.ir(projs)
            )
            loss = self.mse(
                x_iter,
                gts
            )
            loss.backward()
            self.optimizer.step()
            if self.i_iter % self.SHOW_EVERY == 0:
                print('loss', loss.item())
            if self.i_iter % self.SAVE_EVERY == 0:
                self._eval(test_loader)
                self._save('iter_{}.pth'.format(self.i_iter))

    def _eval(self, test_loader):
        self.net.eval()
        rmse_list = []
        ssim_list = []
        psnr_list = []
        for projs, gts in tqdm(test_loader):
            projs = projs.type(self.DTYPE)
            gts = gts.type(self.DTYPE)
            x_iter = self.net(
                self.ir(projs)
            )
            
            for i in range(projs.shape[0]):
                gt = torch_to_np(gts[i:i+1])
                x_iter_npy = np.clip(torch_to_np(x_iter[i:i+1]), 0, 1).astype(np.float64)

                rmse_list.append(mean_squared_error(x_iter_npy, gt))
                ssim_list.append(structural_similarity(x_iter_npy, gt, multichannel=False))
                psnr_list.append(peak_signal_noise_ratio(x_iter_npy, gt))
        print('EVAL_RESULT {}/{}- psnr: {:.3f} - ssim: {:.3f} - rmse: {:.5f}'.format(
                self.name, self.i_iter, np.mean(psnr_list), np.mean(ssim_list), np.mean(rmse_list),
            ))
            


