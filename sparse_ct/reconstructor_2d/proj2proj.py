"""Proj2Proj: self-supervised reconstruction using masked projection loss."""

from pathlib import Path

import numpy as np
import torch
from skimage.metrics import (
    mean_squared_error,
    peak_signal_noise_ratio,
    structural_similarity,
)
from tqdm import tqdm

from sparse_ct.model.dncnn import DnCNN
from sparse_ct.model.skip import Skip
from sparse_ct.model.unet import UNet
from sparse_ct.radon import IRadon, LearnableFilter, Radon
from sparse_ct.tool import np_to_torch, plot_grid, torch_to_np

from .base import Reconstructor
from .mask import Masker


class Proj2ProjReconstructor(Reconstructor):
    """Mask sinograms, reconstruct with FBP and a CNN, then compare held-out projections.

    Angles are in degrees. Sinograms have shape (detectors, angles), or
    (batch, 1, detectors, angles) when passed through a training loader.
    Ground truth is optional and is used only for reporting image metrics.
    """

    INPUT_DEPTH = 1
    IMAGE_DEPTH = 1
    SHOW_EVERY = 50
    SAVE_EVERY = 1000

    def __init__(
        self,
        name="Proj2Proj",
        net="unet",
        lr=0.001,
        n_iter=8000,
        weights=None,
        self_supervised=True,
        learnable_filter=False,
        device=None,
        image_size=512,
    ):
        super().__init__(name)
        if net not in {"skip", "unet", "skipV2", "dncnn"}:
            raise ValueError(f"Unknown network: {net}")
        if n_iter < 1 or image_size < 8 or lr <= 0:
            raise ValueError(
                "n_iter and lr must be positive, and image_size must be at least 8"
            )
        self.device = torch.device(
            device or ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.IMAGE_SIZE = image_size
        self.net_type = net
        self.lr = lr
        self.n_iter = n_iter
        self.weights = weights
        self.selfsupervised = self_supervised
        self.learnable_filter = learnable_filter
        self.gt = None
        self.noisy = None
        self.FOCUS = None
        self.log_dir = None
        self.best_result = None
        self.loss_hist = []
        self.rmse_hist = []
        self.ssim_hist = []
        self.psnr_hist = []

    def set_for_metric(self, gt, noisy, FOCUS=None, log_dir=None):
        self.gt = np.asarray(gt)
        self.noisy = noisy
        self.FOCUS = FOCUS
        self.log_dir = log_dir

    def init_train(self, theta):
        theta = np.asarray(theta, dtype=np.float32)
        if theta.ndim != 1 or len(theta) < 2 or not np.isfinite(theta).all():
            raise ValueError("theta must contain at least two finite angles in degrees")
        self.theta = theta
        self.n_proj = len(theta)
        self.loss_hist = []
        self.rmse_hist = []
        self.ssim_hist = []
        self.psnr_hist = []
        self.best_result = None
        self.i_iter = 0
        self.net = self._get_net(self.net_type)
        self.r = Radon(self.IMAGE_SIZE, theta, True).to(self.device)
        parameters = list(self.net.parameters())
        if self.learnable_filter:
            self.filter = LearnableFilter(self.IMAGE_SIZE).to(self.device)
            self.ir = IRadon(self.IMAGE_SIZE, theta, True, use_filter=self.filter).to(
                self.device
            )
            parameters += list(self.filter.parameters())
        else:
            self.ir = IRadon(self.IMAGE_SIZE, theta, True).to(self.device)
        self.optimizer = torch.optim.Adam(parameters, lr=self.lr)
        self.masker = Masker(width=4, mode="interpolate")
        self.mse = torch.nn.MSELoss().to(self.device)
        if self.weights:
            self._load(self.weights)

    def _to_tensor(self, projections):
        return torch.as_tensor(projections, dtype=torch.float32, device=self.device)

    def _set_training(self, training):
        self.net.train(training)
        self.ir.train(training)

    def _train_step(self, projections):
        self._set_training(True)
        self.optimizer.zero_grad()
        masked, mask = self.masker.mask(projections, self.i_iter % self.masker.n_masks)
        reconstruction = self.net(self.ir(masked))
        loss = self.mse(self.r(reconstruction) * mask, projections * mask)
        loss.backward()
        self.optimizer.step()
        self.i_iter += 1
        return loss.item()

    @torch.no_grad()
    def _reconstruct(self, projections):
        self._set_training(False)
        return self.net(self.ir(projections))

    def _record_metrics(self, image):
        if self.gt is None:
            return
        self.rmse_hist.append(mean_squared_error(self.gt, image))
        self.ssim_hist.append(structural_similarity(self.gt, image, data_range=1.0))
        self.psnr_hist.append(peak_signal_noise_ratio(self.gt, image, data_range=1.0))
        if self.log_dir:
            Path(self.log_dir).mkdir(parents=True, exist_ok=True)
            plot_grid(
                [self.gt, image],
                self.FOCUS,
                save_name=str(Path(self.log_dir) / f"{self.i_iter}.png"),
            )

    def calc(self, projs, theta):
        projs = np.asarray(projs)
        if projs.shape != (self.IMAGE_SIZE, len(theta)) or not np.isfinite(projs).all():
            raise ValueError(
                "projs must be a finite (image_size, number_of_angles) sinogram"
            )
        self.init_train(theta)
        projections = self._to_tensor(np_to_torch(projs))
        if self.selfsupervised:
            for _ in tqdm(range(self.n_iter), desc=self.name):
                self.loss_hist.append(self._train_step(projections))
                if self.i_iter % self.SHOW_EVERY == 0:
                    image = np.clip(torch_to_np(self._reconstruct(projections)), 0, 1)
                    self._record_metrics(image)
        self.image_r = np.clip(
            torch_to_np(self._reconstruct(projections)), 0, 1
        ).astype(np.float64)
        self.best_result = self.image_r.copy()
        return self.image_r

    def _get_net(self, net):
        if net == "skip":
            return Skip(
                num_input_channels=self.INPUT_DEPTH,
                num_output_channels=self.IMAGE_DEPTH,
                upsample_mode="nearest",
                num_channels_down=[16, 32, 64, 128, 256],
                num_channels_up=[16, 32, 64, 128, 256],
            ).to(self.device)
        elif net == "skipV2":
            return Skip(
                num_input_channels=self.INPUT_DEPTH,
                num_output_channels=self.IMAGE_DEPTH,
                upsample_mode="nearest",
                num_channels_down=[32, 64, 128, 256, 512],
                num_channels_up=[32, 64, 128, 256, 512],
            ).to(self.device)
        elif net == "unet":
            return UNet(
                num_input_channels=self.INPUT_DEPTH,
                num_output_channels=self.IMAGE_DEPTH,
                feature_scale=4,
                more_layers=0,
                concat_x=False,
                upsample_mode="bilinear",
                norm_layer=torch.nn.BatchNorm2d,
                pad="reflect",
                need_sigmoid=False,
                need_bias=True,
            ).to(self.device)
        elif net == "dncnn":
            return DnCNN(
                in_channels=self.INPUT_DEPTH,
                out_channels=self.IMAGE_DEPTH,
                num_of_layers=6,
            ).to(self.device)
        raise ValueError(f"Unknown network: {net}")

    def _save(self, save_name):
        save_name = Path(save_name)
        save_name.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.net.state_dict(), save_name)
        if self.learnable_filter:
            torch.save(self.filter.state_dict(), str(save_name) + ".filter")

    def _load(self, load_name):
        self.net.load_state_dict(
            torch.load(load_name, map_location=self.device, weights_only=True)
        )
        if self.learnable_filter:
            self.filter.load_state_dict(
                torch.load(
                    str(load_name) + ".filter",
                    map_location=self.device,
                    weights_only=True,
                )
            )

    def train(self, train_loader, test_loader=None, epochs=10):
        """Train after init_train(theta); validation images are optional."""
        for _ in range(epochs):
            self._train_one_epoch(train_loader, test_loader)
            if test_loader is not None:
                self._eval(test_loader)

    def _train_one_epoch(self, train_loader, test_loader=None):
        for projections in tqdm(train_loader, desc=self.name):
            self.loss_hist.append(self._train_step(self._to_tensor(projections)))
            if self.i_iter % self.SAVE_EVERY == 0:
                if test_loader is not None:
                    self._eval(test_loader)
                self._save(f"iter_{self.i_iter}.pth")

    @torch.no_grad()
    def _eval(self, test_loader):
        metrics = []
        try:
            for projections, ground_truth in tqdm(test_loader, desc="Validation"):
                images = self._reconstruct(self._to_tensor(projections))
                for image, target in zip(images, ground_truth):
                    image = np.clip(image[0].cpu().numpy(), 0, 1)
                    target = torch.as_tensor(target)[0].cpu().numpy()
                    metrics.append(
                        (
                            mean_squared_error(target, image),
                            peak_signal_noise_ratio(target, image, data_range=1.0),
                            structural_similarity(target, image, data_range=1.0),
                        )
                    )
        finally:
            self._set_training(True)
        if not metrics:
            raise ValueError("Validation loader is empty")
        mse, psnr, ssim = np.mean(metrics, axis=0)
        print(f"{self.name}: MSE={mse:.5f} PSNR={psnr:.3f} SSIM={ssim:.3f}")
        return mse, psnr, ssim
