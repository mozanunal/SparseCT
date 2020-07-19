
import torch
from sparse_ct.loss import tv
from .base import BaseDenoiser

class TVDenoiser(BaseDenoiser):
    def __init__(self, name, 
                n_tv_iter, beta=0.2 ):
        super(TVDenoiser, self).__init__(name)
        self.n_iter = n_tv_iter
        self.loss = tv.tv_2d_l2
    
    def calc(self):
        for i in self.n_iter: