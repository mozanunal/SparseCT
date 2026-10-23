import subprocess
import sys

import numpy as np
import pytest
import torch
from skimage.data import shepp_logan_phantom
from skimage.transform import radon, resize

from sparse_ct.data import create_circular_mask
from sparse_ct.radon import IRadon, LearnableFilter, RampFilter
from sparse_ct.reconstructor_2d import N2SelfReconstructor, Proj2ProjReconstructor
from sparse_ct.reconstructor_2d.dataset import DeepLesionDataset
from sparse_ct.reconstructor_2d.mask import Masker


@pytest.fixture(autouse=True)
def deterministic_cpu():
    torch.manual_seed(7)
    torch.set_num_threads(2)


@pytest.fixture
def projections():
    image = resize(shepp_logan_phantom(), (16, 16)) * create_circular_mask(16, 16)
    angles = np.linspace(0, 180, 8, endpoint=False)
    return image, radon(image, theta=angles), angles


def test_proj2proj_import_does_not_load_optional_baselines():
    subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from sparse_ct.reconstructor_2d import Proj2ProjReconstructor; "
                "assert not {'bm3d', 'dival', 'torchvision'} & sys.modules.keys()"
            ),
        ],
        check=True,
    )


def test_grid_masks_partition_sinogram_without_target_leakage():
    masker = Masker(width=4, mode="interpolate")
    projections = torch.randn(2, 1, 17, 9)
    masks = []
    for phase in range(16):
        masked, mask = masker.mask(projections, phase)
        perturbed = projections + 100 * mask
        masked_perturbed, _ = masker.mask(perturbed, phase)
        torch.testing.assert_close(masked * mask, masked_perturbed * mask)
        torch.testing.assert_close(masked * (1 - mask), projections * (1 - mask))
        masks.append(mask)
    torch.testing.assert_close(torch.stack(masks).sum(0), torch.ones(17, 9))


@pytest.mark.parametrize("learnable", [False, True])
def test_filter_matches_numpy_fft_and_backpropagates(learnable):
    projections = torch.randn(2, 1, 17, 8, requires_grad=True)
    values = projections.detach().numpy()
    if learnable:
        filter_module = LearnableFilter(17)
        expected = np.fft.ifft(
            np.fft.fft(values, axis=-2) * (2 * np.abs(np.fft.fftfreq(17)))[:, None],
            axis=-2,
        ).real
    else:
        filter_module = RampFilter()
        impulse = np.zeros(64)
        impulse[0] = 0.25
        odd_indices = np.concatenate((np.arange(1, 33, 2), np.arange(31, 0, -2)))
        impulse[1::2] = -1 / (np.pi * odd_indices) ** 2
        padded = np.pad(values, ((0, 0), (0, 0), (0, 47), (0, 0)))
        expected = np.fft.ifft(
            np.fft.fft(padded, axis=-2) * (2 * np.fft.fft(impulse).real)[:, None],
            axis=-2,
        ).real[..., :17, :]
    result = filter_module(projections)
    np.testing.assert_allclose(result.detach().numpy(), expected, atol=3e-7)
    result.square().mean().backward()
    assert torch.isfinite(projections.grad).all()
    assert projections.grad.abs().sum() > 0
    if learnable:
        assert filter_module.filter.grad.abs().sum() > 0


def test_fbp_remains_differentiable(projections):
    _, sinogram, angles = projections
    inputs = torch.tensor(sinogram[None, None], dtype=torch.float32, requires_grad=True)
    image = IRadon(16, angles)(inputs)
    assert image.shape == (1, 1, 16, 16)
    assert torch.isfinite(image).all()
    image.square().mean().backward()
    assert torch.isfinite(inputs.grad).all()
    assert inputs.grad.abs().sum() > 0


def test_single_image_training_uses_all_masks_without_reference_images(
    projections, monkeypatch
):
    _, sinogram, angles = projections
    phases = []
    original_mask = Masker.mask

    def record_mask(self, inputs, phase):
        phases.append(phase)
        return original_mask(self, inputs, phase)

    monkeypatch.setattr(Masker, "mask", record_mask)
    reconstructor = Proj2ProjReconstructor(
        net="dncnn", image_size=16, n_iter=16, device="cpu"
    )
    reconstructor.SHOW_EVERY = 3
    image = reconstructor.calc(sinogram, angles)
    assert phases == list(range(16))
    assert reconstructor.i_iter == 16
    assert image.shape == (16, 16)
    assert np.isfinite(image).all()
    assert image.min() >= 0 and image.max() <= 1
    assert np.isfinite(reconstructor.loss_hist).all()
    assert len(reconstructor.loss_hist) == 16
    assert reconstructor.gt is None
    assert reconstructor.log_dir is None


@pytest.mark.parametrize("learnable", [False, True])
def test_checkpoint_round_trip_and_inference_has_no_optimizer_steps(
    projections, tmp_path, learnable
):
    _, sinogram, angles = projections
    trainer = Proj2ProjReconstructor(
        net="dncnn",
        image_size=16,
        n_iter=2,
        device="cpu",
        learnable_filter=learnable,
    )
    expected = trainer.calc(sinogram, angles)
    checkpoint = tmp_path / "nested" / "proj2proj.pth"
    trainer._save(checkpoint)
    inference = Proj2ProjReconstructor(
        net="dncnn",
        image_size=16,
        weights=checkpoint,
        self_supervised=False,
        learnable_filter=learnable,
        device="cpu",
    )
    actual = inference.calc(sinogram, angles)
    np.testing.assert_array_equal(actual, expected)
    assert inference.i_iter == 0
    assert not inference.loss_hist


def test_validation_preserves_parameters_and_batchnorm_buffers(projections):
    image, sinogram, angles = projections
    reconstructor = Proj2ProjReconstructor(net="dncnn", image_size=16, device="cpu")
    reconstructor.init_train(angles)
    before = {
        key: value.clone() for key, value in reconstructor.net.state_dict().items()
    }
    batch = torch.tensor(sinogram[None, None], dtype=torch.float32)
    targets = torch.tensor(image[None, None], dtype=torch.float32)
    metrics = reconstructor._eval([(batch, targets)])
    assert np.isfinite(metrics).all()
    assert reconstructor.i_iter == 0
    assert not reconstructor.optimizer.state
    for key, value in reconstructor.net.state_dict().items():
        torch.testing.assert_close(value, before[key], rtol=0, atol=0)
    reconstructor.train([batch], [(batch, targets)], epochs=2)
    assert reconstructor.i_iter == 2
    assert reconstructor.optimizer.state
    assert any(
        not torch.equal(value, before[key])
        for key, value in reconstructor.net.state_dict().items()
    )


def test_optional_metrics_work_with_float_images(projections):
    gt, sinogram, angles = projections
    reconstructor = Proj2ProjReconstructor(
        net="dncnn", image_size=16, n_iter=1, device="cpu"
    )
    reconstructor.SHOW_EVERY = 1
    reconstructor.set_for_metric(gt, gt)
    reconstructor.calc(sinogram, angles)
    assert len(reconstructor.ssim_hist) == 1
    assert np.isfinite(reconstructor.eval(gt)).all()


def test_legacy_constructor_maps_original_options():
    with pytest.warns(DeprecationWarning, match="Proj2ProjReconstructor"):
        legacy = N2SelfReconstructor(
            "legacy",
            n2self_n_iter=5,
            n2self_weights="old.pth",
            n2self_selfsupervised=False,
            n2self_proj_ratio=0.4,
            device="cpu",
        )
    assert isinstance(legacy, Proj2ProjReconstructor)
    assert legacy.net_type == "skip"
    assert legacy.n_iter == 5
    assert legacy.weights == "old.pth"
    assert not legacy.selfsupervised


def test_image_training_dataset_does_not_require_dival():
    from pathlib import Path

    import sparse_ct

    image_path = Path(sparse_ct.__file__).parent / "data/benchmark_human/4.png"
    dataset = DeepLesionDataset([image_path], img_size=16, n_proj=8)
    projections = dataset[0]
    assert projections.shape == (1, 16, 8)
    assert np.isfinite(projections).all()


def test_default_unet_can_train_on_cpu():
    angles = np.linspace(0, 180, 8, endpoint=False)
    reconstructor = Proj2ProjReconstructor(image_size=64, n_iter=1, device="cpu")
    image = reconstructor.calc(np.random.default_rng(7).random((64, 8)), angles)
    assert image.shape == (64, 64)
    assert np.isfinite(image).all()
    assert reconstructor.i_iter == 1
