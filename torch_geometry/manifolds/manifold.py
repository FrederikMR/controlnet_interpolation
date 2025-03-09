#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:54:30 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch
from torch.func import jacfwd
from torch import Tensor

from typing import Callable

from abc import ABC

#%% Riemannian Manifold

class RiemannianManifold(ABC):
    def __init__(self,
                 G:Callable[[Tensor], Tensor]=None,
                 f:Callable[[Tensor], Tensor]=None,
                 invf:Callable[[Tensor],Tensor]=None,
                 )->None:
        
        self.f = f
        self.invf = invf
        if ((G is None) and (f is None)):
            raise ValueError("Both the metric, g, and chart, f, is not defined")
        elif (G is None):
            self.G = lambda z: self.pull_back_metric(z)
        else:
            self.G = G
            
        return
        
    def __str__(self)->str:
        
        return "Riemannian Manifold base object"
    
    def Jf(self,
           z:Tensor,
           )->Tensor:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            return jacfwd(self.f)(z)
        
    def pull_back_metric(self,
                         z:Tensor
                         )->Tensor:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            Jf = self.Jf(z)
            return torch.einsum('...ik,...il->...kl', Jf, Jf)
    
    def DG(self,
           z:Tensor
           )->Tensor:

        return jacfwd(self.G)(z)
    
    def Ginv(self,
             z:Tensor
             )->Tensor:
        
        return torch.linalg.inv(self.G(z))
    
    def christoffel_symbols(self,
                            z:Tensor
                            )->Tensor:
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        
        return 0.5*(torch.einsum('im,kml->ikl',gsharpx,Dgx)
                   +torch.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -torch.einsum('im,klm->ikl',gsharpx,Dgx))
    
    def geodesic_equation(self,
                          z:Tensor,
                          v:Tensor
                          )->Tensor:
        
        Gamma = self.Chris(z)

        dx1t = v
        dx2t = -torch.einsum('ikl,k,l->i',Gamma,v,v)
        
        return torch.hstack((dx1t,dx2t))
    
    def energy(self, 
               gamma:Tensor,
               )->Tensor:
        
        T = len(gamma)
        
        G = self.G(gamma[:-1])
        dz = gamma[1:]-gamma[:-1]
        energy = torch.einsum('...i,...ij,...j->...', dz, G, dz)
        
        return T*torch.sum(energy)
    
    def length(self, 
               gamma:Tensor,
               )->Tensor:

        G = self.G(gamma[:-1])
        dz = gamma[1:]-gamma[:-1]
        length = torch.sqrt(torch.einsum('...i,...ij,...j->...', dz, G, dz))
        
        return torch.sum(length)