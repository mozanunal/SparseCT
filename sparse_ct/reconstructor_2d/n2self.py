"""Compatibility API. New code should use Proj2ProjReconstructor."""

import warnings

from .proj2proj import Proj2ProjReconstructor


class N2SelfReconstructor(Proj2ProjReconstructor):
    def __init__(
        self,
        name,
        net="skip",
        lr=0.001,
        n2self_n_iter=8000,
        n2self_weights=None,
        n2self_selfsupervised=True,
        n2self_proj_ratio=0.2,
        learnable_filter=False,
        **kwargs,
    ):
        warnings.warn(
            "N2SelfReconstructor is now Proj2ProjReconstructor; use n_iter, weights, "
            "and self_supervised instead of the n2self_ options. "
            "n2self_proj_ratio is unused by grid masking.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(
            name,
            net=net,
            lr=lr,
            n_iter=n2self_n_iter,
            weights=n2self_weights,
            self_supervised=n2self_selfsupervised,
            learnable_filter=learnable_filter,
            **kwargs,
        )
        self.n2self_proj_ratio = n2self_proj_ratio
