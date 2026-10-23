"""Train Proj2Proj from image files with simulated noisy projections."""

import argparse
from pathlib import Path

import numpy as np
from torch.utils.data import DataLoader

from sparse_ct.reconstructor_2d import Proj2ProjReconstructor
from sparse_ct.reconstructor_2d.dataset import DeepLesionDataset


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--validation-dir", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("checkpoints/proj2proj.pth")
    )
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--views", type=int, default=64)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    train_files = sorted(args.train_dir.rglob("*.png"))
    if not train_files:
        parser.error("--train-dir must contain PNG images")
    train_loader = DataLoader(
        DeepLesionDataset(train_files, n_proj=args.views),
        batch_size=args.batch_size,
        shuffle=True,
    )
    validation_loader = None
    if args.validation_dir:
        validation_files = sorted(args.validation_dir.rglob("*.png"))
        if not validation_files:
            parser.error("--validation-dir must contain PNG images")
        validation_loader = DataLoader(
            DeepLesionDataset(validation_files, return_gt=True, n_proj=args.views),
            batch_size=args.batch_size,
        )
    reconstructor = Proj2ProjReconstructor(device=args.device)
    reconstructor.init_train(np.linspace(0, 180, args.views, endpoint=False))
    reconstructor.train(train_loader, validation_loader, epochs=args.epochs)
    reconstructor._save(args.output)


if __name__ == "__main__":
    main()
