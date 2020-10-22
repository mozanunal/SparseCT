
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

from sparse_ct.data import (image_to_sparse_sinogram, 
                        ellipses_to_sparse_sinogram)
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

class Dataset(torch.utils.data.Dataset):
    def __init__(self, file_list, 
                return_gt=False,
                n_proj=32,
                noise_pow=25.0,
                img_size=512):
        self.file_list = file_list
        self.return_gt = return_gt
        self.noise_pow = noise_pow
        self.n_proj = n_proj
        self.size = img_size

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        # get image
        gt, sinogram, _, _ = image_to_sparse_sinogram(
            self.file_list[index],
            gray=True,
            n_proj=self.n_proj,
            channel=1,
            size=self.size,
            noise_pow=self.noise_pow
            )
        # get projs
        if self.return_gt:
            return np.expand_dims(sinogram, axis=0), np.expand_dims(gt, axis=0)
        else:
            return np.expand_dims(sinogram, axis=0)      

class EllipsesDataset(torch.utils.data.Dataset):
    def __init__(self, ellipses_type, 
                return_gt=False,
                n_proj=32,
                noise_pow=25.0,
                img_size=512):
        self.ellipses_type = ellipses_type
        self.return_gt = return_gt
        self.noise_pow = noise_pow
        self.n_proj = n_proj
        self.size = img_size

    def __len__(self):
        if self.ellipses_type == 'train':
            return 32000
        elif self.ellipses_type == 'validation':
            return 1000

    def __getitem__(self, index):
        # get image
        gt, sinogram, _, _ = ellipses_to_sparse_sinogram(
            part=self.ellipses_type,
            gray=True,
            n_proj=self.n_proj,
            channel=1,
            size=self.size,
            noise_pow=self.noise_pow
            )
        # get projs
        if self.return_gt:
            return np.expand_dims(sinogram, axis=0), np.expand_dims(gt, axis=0)
        else:
            return np.expand_dims(sinogram, axis=0)


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
        n2self_proj_ratio=0.2):
        super(N2SelfReconstructor, self).__init__(name)
        self.n_iter = n2self_n_iter
        self.n2self_proj_ratio = n2self_proj_ratio
        assert net in ['skip', 'unet', 'skipV2']
        self.lr = lr
        
        # net
        self.net_type = net
        self.weights = n2self_weights
        self.selfsupervised = n2self_selfsupervised

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
            
        projs_torch = np_to_torch(projs).type(self.DTYPE)
        norm = transforms.Normalize(projs_torch[0].mean((1,2)), projs_torch[0].std((1,2)))
        full = set(i for i in range(self.n_proj))

        for i in tqdm(range(self.n_iter)):
            
            # iter
            self.optimizer.zero_grad()

            # train
            if i % self.SHOW_EVERY != 0:
                idx, idxC = toSubLists(full, self.n2self_proj_ratio)
                tsub = self.theta[idx]
                tsubC = self.theta[idxC]
            # val
            else:
                idx, idxC = list(full), list(full)
                tsub = self.theta
                tsubC = self.theta

            r = Radon(self.IMAGE_SIZE, tsub, True).to(self.DEVICE)
            ir = IRadon(self.IMAGE_SIZE, tsubC, True).to(self.DEVICE)

            x_iter = self.net(
                ir(projs_torch[:,:,:,idxC])
            )
            loss = self.mse(
                # norm(r(x_iter)[0]).unsqueeze(0),
                # norm(projs_torch[:,:,:,idx][0]).unsqueeze(0)
                r(x_iter),
                projs_torch[:,:,:,idx]
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
                plot_grid([self.gt, x_iter_npy], self.FOCUS, save_name=self.log_dir+'/{}.png'.format(i))

        self.image_r = self.best_result
        return self.image_r
    
    def _calc_supervised(self, projs, theta):
        self.net.eval()
        projs = np_to_torch(projs).type(self.DTYPE)
        ir = IRadon(self.IMAGE_SIZE, self.theta, True).to(self.DEVICE)
        x_iter = self.net(
            ir(projs)
        )
        x_iter_npy = np.clip(torch_to_np(x_iter), 0, 1).astype(np.float64)
        self.image_r = x_iter_npy.copy()
        return self.image_r

    def init_train(self, theta):
        self.theta = torch.from_numpy(theta).type(self.DTYPE)
        self.n_proj = len(theta)
        self.net = self._get_net(self.net_type)
        if self.weights:
            self._load(self.weights)
        self.mse = torch.nn.MSELoss().to(self.DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.i_iter = 0

    def calc(self, projs, theta):

        # reinit everything
        self.theta = torch.from_numpy(theta).type(self.DTYPE)
        self.n_proj = len(theta)
        self.loss_hist = []
        self.rmse_hist = []
        self.ssim_hist = []
        self.psnr_hist = []
        self.net = self._get_net(self.net_type)
        if self.weights:
            self._load(self.weights)
        self.best_result = None
        # loss functions and optimization
        self.mse = torch.nn.MSELoss().to(self.DEVICE)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.lr)
        self.i_iter = 0

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
        full = set(i for i in range(self.n_proj))
        self.weights = True
        self.net.train()
        for projs in tqdm(train_loader):
            self.i_iter += 1

            projs = projs.type(self.DTYPE)

            self.optimizer.zero_grad()

            idx, idxC = toSubLists(full, self.n2self_proj_ratio)
            tsub = self.theta[idx]
            tsubC = self.theta[idxC]
            r = Radon(self.IMAGE_SIZE, tsub, True).to(self.DEVICE)
            ir = IRadon(self.IMAGE_SIZE, tsubC, True).to(self.DEVICE)

            x_iter = self.net(
                ir(projs[:,:,:,idxC])
            )
            loss = self.mse(
                r(x_iter),
                projs[:,:,:,idx]
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
            ir = IRadon(self.IMAGE_SIZE, self.theta, True).to(self.DEVICE)
            x_iter = self.net(
                ir(projs)
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
            


