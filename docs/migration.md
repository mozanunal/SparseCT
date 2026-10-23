# Proj2Proj migration

The canonical implementation is now `sparse_ct.reconstructor_2d.proj2proj` and the
public class is `Proj2ProjReconstructor`.

```python
from sparse_ct.reconstructor_2d import Proj2ProjReconstructor

reconstructor = Proj2ProjReconstructor(
    "Proj2Proj", net="unet", n_iter=4000, weights=None, self_supervised=True,
)
```

## Old and new names

| Old argument | New argument |
| :--- | :--- |
| `n2self_n_iter` | `n_iter` |
| `n2self_weights` | `weights` |
| `n2self_selfsupervised` | `self_supervised` |
| `n2self_proj_ratio` | Removed from the new API; grid masking never used this value |

The old `N2SelfReconstructor` import and constructor signature remain as a
compatibility wrapper. Its default network remains `skip`; the new API defaults
to `unet`, as used by Proj2Proj. Existing network state dictionaries remain
loadable when the architecture matches. Learnable filter checkpoints retain the
`filter` parameter name and shape in the adjacent `.filter` file.

## Runtime changes

- CPU and CUDA are supported by the Proj2Proj path. `image_size` is configurable.
- Filters use `torch.fft` in place of the backend's removed `torch.rfft` API.
- All 16 positions of the 4 x 4 mask are used, with the mask index advanced after
  each optimizer update. The previous dataset loop used modulo 15, while the
  single-image loop did not advance its mask index.
- Every optimizer update uses masked projections. Periodic full-sinogram
  inference and validation run without gradient tracking or optimizer updates.
- Reconstruction works without reference images or a log directory.
- `calc` returns the final full-sinogram reconstruction. The old implementation
  selected a periodic result using an unmasked loss and updated weights during
  those evaluation iterations. `best_result` now mirrors the final output for
  compatibility. These fixes can change numerical results from historical runs.
- Floating-point SSIM and PSNR metrics use an explicit data range of 1.0.
- `train` now runs the requested epochs after `init_train(theta)`. Validation is
  optional and does not contribute to optimization.

This is a maintained execution path, not a reproduction of the paper's reported
results. Dataset splits, training duration, checkpoints, and random seeds must be
matched separately for a reproduction study.

## Historical variants

`n2selfS.py` contains an angle split experiment. It remains under its original
name to preserve historical imports, and is not the canonical Proj2Proj path.
The `papers/` scripts retain their names and dataset assumptions. DGR and the
supervised methods are separate algorithms, not aliases for Proj2Proj.
