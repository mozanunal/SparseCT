# Historical paper experiments

The maintained Proj2Proj implementation is
[`proj2proj.py`](../sparse_ct/reconstructor_2d/proj2proj.py). Start with the
[Proj2Proj example](../sparse_ct/example/proj2proj_example.py) and
[training script](../sparse_ct/example/proj2proj_train.py).

- `self_super_ct_reconstuction/` contains scripts for the 2021 ISBI work that
  preceded Proj2Proj. The directory spelling, N2Self working names, dataset paths,
  and checkpoint names are preserved as historical context.
- `dgr/` contains experiments for the separate deep generative regularization
  paper. DGR is not the Proj2Proj method.

These scripts are not portable examples and are not covered by the modern CPU
checks. They assume external datasets, checkpoints, and the original research
setup. The original dependency pins are recorded in `requirements-legacy.txt`
and `requirements-dump.txt` at the repository root.
