"""Reconstruct a bundled image using Proj2Proj without external data or weights."""

import argparse
from pathlib import Path

import numpy as np
from skimage.transform import iradon

from sparse_ct.data import image_to_sparse_sinogram
from sparse_ct.reconstructor_2d import Proj2ProjReconstructor
from sparse_ct.tool import plot_grid


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image", type=Path, default=Path(__file__).parents[1] / "data/ct1.jpg"
    )
    parser.add_argument("--output", type=Path, default=Path("results/proj2proj.png"))
    parser.add_argument("--iterations", type=int, default=100)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--views", type=int, default=64)
    parser.add_argument(
        "--net", choices=["unet", "skip", "skipV2", "dncnn"], default="unet"
    )
    parser.add_argument("--device", default=None)
    parser.add_argument("--weights", type=Path)
    parser.add_argument("--inference-only", action="store_true")
    args = parser.parse_args()
    if args.inference_only and args.weights is None:
        parser.error("--inference-only requires --weights")

    gt, sinogram, angles, _ = image_to_sparse_sinogram(
        args.image,
        n_proj=args.views,
        size=args.size,
        noise_pow=33.0,
    )
    reconstructor = Proj2ProjReconstructor(
        net=args.net,
        n_iter=args.iterations,
        weights=args.weights,
        self_supervised=not args.inference_only,
        device=args.device,
        image_size=args.size,
    )
    reconstruction = reconstructor.calc(sinogram, angles)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output.with_suffix(".npy"), reconstruction)
    plot_grid(
        [gt, iradon(sinogram, theta=angles), reconstruction], save_name=str(args.output)
    )
    mse, psnr, ssim = reconstructor.eval(gt)
    print(f"Proj2Proj: MSE={mse:.5f} PSNR={psnr:.3f} SSIM={ssim:.3f}")
    print(f"Saved ground truth, FBP, Proj2Proj comparison to {args.output}")


if __name__ == "__main__":
    main()
