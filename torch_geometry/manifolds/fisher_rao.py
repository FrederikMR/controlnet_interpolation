#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 10:51:29 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch

from torch import Tensor, vmap
from torch.func import jacrev

from typing import Callable

####################

from .manifold import RiemannianManifold

#%% Code

class FisherRao(RiemannianManifold):
    def __init__(self,
                 log_prob:Callable[[torch.Tensor], torch.Tensor],
                 dim:int=2,
                 )->None:

        self.dim = dim
        hessian = self.make_hessian(log_prob)

        super().__init__(G=lambda x: -hessian(x), f=lambda x: x, invf= lambda x: x)
        
        return
    
    def __str__(self)->str:
        
        return f"Euclidean manifold of dimension {self.dim} in standard coordinates"
    
    def make_hessian(self, log_prob):
        """
        Given log_prob: (d,) -> scalar,
        return a function h(x): (..., d) -> (..., d, d)
        """
        grad_fn = jacrev(log_prob)         # (d,) -> (d,)
        hess_fn = jacrev(grad_fn)          # (d,) -> (d, d)
    
        # vmap over *all* leading dimensions
        # so input (..., d) -> output (..., d, d)
        def hessian_batched(x):
            # x shape: (..., d)
            in_dims = [0] * (x.ndim - 1) + [None]  # but easier: use vmap nesting
            f = hess_fn
            # Wrap with vmap once for each batch dim
            for _ in range(x.ndim - 1):
                f = vmap(f)
            return f(x)
    
        return hessian_batched
    
    def metric(self,
               z:Tensor,
               )->Tensor:
        
        if z.ndim == 1:
            return torch.eye(self.dim)
        else:            
            diag = torch.eye(self.dim)
        
            diag_multi = z.unsqueeze(2).expand(*z.size(), z.size(1))
            diag3d = diag_multi*diag
        
            return diag3d
    
    def dist(self,
             z1:Tensor,
             z2:Tensor
             )->Tensor:
        
        return torch.linalg.norm(z2-z1, axis=-1)
    
    def Geodesic(self,
                 x:Tensor,
                 y:Tensor,
                 t_grid:Tensor=None,
                 )->Tensor:
        
        if t_grid is None:
            t_grid = torch.linspace(0.,1.,100)
        
        return x+(y-x)*t_grid.reshape(-1,1)
    