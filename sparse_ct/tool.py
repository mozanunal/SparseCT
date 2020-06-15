import numpy as np
import torch


def im2tensor(x, grad=False):
    t = torch.from_numpy(x).float()#.to(device)
    t.requires_grad = grad
    return t