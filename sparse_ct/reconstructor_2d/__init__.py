"""Reconstruction methods, imported on demand to isolate optional dependencies."""

from importlib import import_module

_METHOD_MODULES = {
    "Proj2ProjReconstructor": "proj2proj",
    "N2SelfReconstructor": "n2self",
    "IRadonReconstructor": "analytic",
    "FBP_BM3DReconstructor": "analytic",
    "SartReconstructor": "iterative",
    "SartTVReconstructor": "iterative",
    "SartBM3DReconstructor": "iterative",
    "SinBM3DReconstructor": "iterative",
    "DgrReconstructor": "dgr",
    "SupervisedReconstructor": "supervised",
    "SupervisedItReconstructor": "supervised_iterative",
}

__all__ = list(_METHOD_MODULES)


def __getattr__(name):
    if name not in _METHOD_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(f".{_METHOD_MODULES[name]}", __name__)
    method = getattr(module, name)
    globals()[name] = method
    return method
