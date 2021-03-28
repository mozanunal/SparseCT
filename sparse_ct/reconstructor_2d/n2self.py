
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
from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from sparse_ct.model.dncnn import DnCNN
from sparse_ct.model.unet import UNet
from sparse_ct.model.skip import Skip
from sparse_ct.tool import im2tensor, plot_grid, np_to_torch, torch_to_np
from .dataset import DeepLesionDataset, EllipsesDataset
from .base import Reconstructor
from .mask import Masker



# def toSubLists(full, ratio):
#     idx = set(
#         random.choices(
#             list(full),k=int(ratio*len(full))
#         )
#     )
#     idxC = list(full-idx)
#     idx = list(idx)
#     return idx, idxC

def toSubLists(full, ratio):
    a = list(full)
    return a[0:][::2], a[1:][::2]

def _FOCUS(img):
    return img[300:450,200:350]


class N2SelfReconstructor(Reconstructor):
    DEVICE = 'cuda'
    DTYPE = torch.cuda.FloatTensor
    INPUT_DEPTH = 1
    IMAGE_DEPTH = 1
    IMAGE_SIZE = 512
    SHOW_EVERY = 50
    SAVE_EVERY = 1000

    def __init__(self, name,
        net='skip', lr=0.001,
        n2self_n_iter=8000, 
        n2self_weights=None,
        n2self_selfsupervised=True,
        n2self_proj_ratio=0.2,
        learnable_filter=False):
        super(N2SelfReconstructor, self).__init__(name)
        self.n_iter = n2self_n_iter
        self.n2self_proj_ratio = n2self_proj_ratio
        assert net in ['skip', 'unet', 'skipV2', 'dncnn']
        self.lr = lr
        
        # net
        self.net_type = net
        self.weights = n2self_weights
        self.selfsupervised = n2self_selfsupervised
        self.learnable_filter = learnable_filter

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

    def _calc_unsupervised(self, projs, theta):
        if not os.path.exists(self.log_dir):
            os.mkdir(self.log_dir)
            
        projs = np_to_torch(projs).type(self.DTYPE)
        #norm = transforms.Normalize(projs_torch[0].mean((1,2)), projs_torch[0].std((1,2)))

        for i in tqdm(range(self.n_iter)):
            
            # iter
            self.optimizer.zero_grad()

            # train
            if i % self.SHOW_EVERY != 0:
                self.net.train()
                if self.learnable_filter:
                    self.filter.train()
                net_input, mask = self.masker.mask( projs, self.i_iter % (self.masker.n_masks - 1) )
                x_iter = self.net(
                    self.ir(net_input)
                )
                loss = self.mse(
                    self.r(x_iter)*mask,
                    projs*mask
                )
            # val
            else:
                self.net.eval()
                if self.learnable_filter:
                    self.filter.eval()
                x_iter = self.net(
                    self.ir(projs)
                )
                loss = self.mse(
                    self.r(x_iter),
                    projs
                )

            loss.backward()
            self.optimizer.step()

            # metric
            if i % self.SHOW_EVERY == 0:
                x_iter_npy = np.clip(torch_to_np(x_iter), 0, 1).astype(np.float64)
                self.rmse_hist.append(
                    mean_squared_error(x_iter_npy, self.gt))
                self.ssim_hist.append(
                    structural_similarity(x_iter_npy, self.gt, multichannel=False)
                )
                self.psnr_hist.append(
                    peak_signal_noise_ratio(x_iter_npy, self.gt)
                )
                self.loss_hist.append(loss.item())
                print('{}/{}- psnr: {:.3f} - ssim: {:.3f} - rmse: {:.5f} - loss: {:.5f} '.format(
                    self.name, i, self.psnr_hist[-1], self.ssim_hist[-1], self.rmse_hist[-1], self.loss_hist[-1]
                ))

                if i > 2:
                    if self.loss_hist[-1] < min(self.loss_hist[0:-1]):
                        # save network
                        # best_network = [x.detach().cpu() for x in net.parameters()]
                        self.best_result = x_iter_npy.copy() 
                else:
                    self.best_result = x_iter_npy.copy()
                plot_grid([self.gt, x_iter_npy], self.FOCUS, save_name=self.log_dir+'/{}.png'.format(i))

        self.image_r = self.best_result
        return self.image_r
    
    def _calc_supervised(self, projs, theta):
        self.net.eval()
        if self.learnable_filter:
            self.filter.eval()
        projs = np_to_torch(projs).type(self.DTYPE)
        x_iter = self.net(
            self.ir(projs)
        )
        x_iter_npy = np.clip(torch_to_np(x_iter), 0, 1).astype(np.float64)
        self.image_r = x_iter_npy.copy()
        return self.image_r

    def init_train(self, theta):
        self.theta = torch.from_numpy(theta).type(self.DTYPE)
        self.n_proj = len(theta)
        self.loss_hist = []
        self.rmse_hist = []
        self.ssim_hist = []
        self.psnr_hist = []
        self.net = self._get_net(self.net_type)
        self.i_iter = 0
        self.r = Radon(self.IMAGE_SIZE, theta, True).to(self.DEVICE)
        if self.learnable_filter:
            self.filter = LearnableFilter(512)
            self.ir = IRadon(self.IMAGE_SIZE, theta, True, use_filter=self.filter).to(self.DEVICE)
            self.optimizer = torch.optim.Adam(list(self.net.parameters())+list(self.filter.parameters()), lr=self.lr)
        else:
            self.ir = IRadon(self.IMAGE_SIZE, theta, True).to(self.DEVICE)
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.masker = Masker(width = 4, mode='interpolate')
        self.mse = torch.nn.MSELoss().to(self.DEVICE)
        self.ssim = MS_SSIM(data_range=1.0, size_average=True, channel=self.IMAGE_DEPTH).to(self.DEVICE)
        if self.weights:
            self._load(self.weights)

    def calc(self, projs, theta):
        
        # reinit everything
        self.init_train(theta)

        if self.selfsupervised:
            return self._calc_unsupervised(projs, theta)
        else:
            return self._calc_supervised(projs, theta)

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
        elif net == 'dncnn':
            return DnCNN(in_channels=self.INPUT_DEPTH, 
                out_channels=self.IMAGE_DEPTH, num_of_layers=6).to(self.DEVICE)
        else:
            assert False



    ######### Training #########
    def _save(self, save_name):
        torch.save(self.net.state_dict(), save_name)
        if self.learnable_filter:
            torch.save(self.filter.state_dict(), save_name+'.filter')

    def _load(self, load_name):
        print('weights are loaded...')
        self.net.load_state_dict(torch.load(load_name))
        if self.learnable_filter:
            self.filter.load_state_dict(torch.load(load_name+'.filter'))

    def train(self, train_loader, test_loader, epochs=10):
        pass

    def _train_one_epoch(self, train_loader, test_loader):
        full = set(i for i in range(self.n_proj))
        self.net.train()
        if self.learnable_filter:
            self.filter.train()

        for projs in tqdm(train_loader):
            self.i_iter += 1

            projs = projs.type(self.DTYPE)

            self.optimizer.zero_grad()            
            net_input, mask = self.masker.mask( projs, self.i_iter % (self.masker.n_masks - 1) )

            x_iter = self.net(
                self.ir(net_input)
            )
            loss = self.mse(
                self.r(x_iter)*mask,
                projs*mask
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
        if self.learnable_filter:
            self.filter.eval()
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
                # print('{}/{}- psnr: {:.3f} - ssim: {:.3f} - rmse: {:.5f}'.format(
                #     self.name, i, psnr_list[-1], ssim_list[-1], rmse_list[-1],
                # ))
        print('EVAL_RESULT {}/{}- psnr: {:.3f} - ssim: {:.3f} - rmse: {:.5f}'.format(
                self.name, self.i_iter, np.mean(psnr_list), np.mean(ssim_list), np.mean(rmse_list),
            ))
            


