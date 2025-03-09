#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb  5 03:28:06 2025

@author: fmry
"""

#%% Modules

import torch
from torch import vmap

from torch import Tensor
from typing import Callable, Dict, Tuple
from abc import ABC

from .utils import GeoCurve
from torch_geometry.manifolds import RiemannianManifold
from torch_geometry.line_search import Backtracking

#%% Testing

class ProbTestGEORCE(ABC):
    def __init__(self,
                 score_fun:Callable,
                 init_fun:Callable=None,
                 lam:float=1.0,
                 T:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 save_step:int=1,
                 save_path:str=None,
                 )->None:

        self.score_fun = score_fun
        
        self.lam = lam
        self.T = T
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        self.save_step = save_step
        self.save_path = save_path
        
        if init_fun is None:
            self.init_fun = lambda z0, zT, T: (zT-z0)*torch.linspace(0.0,
                                                                     1.0,
                                                                     T+1,
                                                                     dtype=z0.dtype)[1:-1].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return "Geodesic Computation Object using Control Problem with Probability Flow"

    @torch.no_grad()
    def score_norm2(self, 
                    zt:Tensor,
                    )->Tensor:
        
        score = self.score_fun(zt)
        score_norm2 = torch.sum(score**2)

        return score_norm2

    @torch.no_grad()
    def energy(self, 
               zt:Tensor,
               )->Tensor:

        zt = torch.vstack((self.z0, zt, self.zT))
        ut = zt[1:]-zt[:-1]
        
        return torch.sum(ut*ut)
    
    @torch.no_grad()
    def reg_energy(self, 
                   zt:Tensor,
                   *args,
                   )->Tensor:

        score_norm2 = self.score_norm2(zt)
        energy = self.energy(zt)
        
        return self.lam_norm*energy+score_norm2
    
    @torch.no_grad()
    def Dregenergy(self,
                   ut:Tensor,
                   gt:Tensor,
                   )->Tensor:
        
        return (gt+2.*(ut[:-1]-ut[1:])).reshape(-1)

    def inner_product(self,
                      zt:Tensor,
                      )->Tensor:

        score = self.score_fun(zt)
        score_norm2 = torch.sum(score**2)
        
        return score_norm2
    
    def gt(self,
           model:GeoCurve,
           )->Tensor:
        
        zt = model.forward()
        loss = self.inner_product(zt)
        loss.backward()
        gt = model.zt.grad

        return gt.detach()
        
    @torch.no_grad()
    def update_xt(self,
                  zt:Tensor,
                  alpha:Tensor,
                  ut_hat:Tensor,
                  ut:Tensor,
                  )->Tensor:
        
        return self.z0+torch.cumsum(alpha*ut_hat[:-1]+(1.-alpha)*ut[:-1], dim=0)
    
    @torch.no_grad()
    def update_ut(self,
                  gt:Tensor,
                  )->Tensor:
        
        g_cumsum = torch.vstack((torch.flip(torch.cumsum(torch.flip(gt, dims=[0]), dim=0), dims=[0]), 
                                 torch.zeros(self.dim)))
        g_sum = torch.sum(g_cumsum, dim=0)/self.T
        
        return self.diff/self.T+0.5*(g_sum-g_cumsum)

    def cond_fun(self, 
                 grad_val:Tensor,
                 idx:int,
                 )->Tensor:
        
        grad_norm = torch.linalg.norm(grad_val.reshape(-1))

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                   model:GeoCurve,
                   gt:Tensor,
                   grad_val:Tensor,
                   idx:int,
                   )->Tensor:
        
        zt = model.zt
        ut = model.ut(zt)
        
        ut_hat = self.update_ut(gt)
        tau = self.line_search(zt, grad_val, ut_hat, ut)

        ut = tau*ut_hat+(1.-tau)*ut
        with torch.no_grad():
            model.zt = torch.nn.Parameter(self.z0+torch.cumsum(ut[:-1], dim=0))
        
        gt = self.gt(model)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_val = self.Dregenergy(ut, gt)
        
        return model, gt, grad_val, idx+1
    
    @torch.no_grad()
    def save_model(self,
                   model:GeoCurve,
                   grad_val:Tensor,
                   idx:int,
                   ):
        
        reg_energy = self.reg_energy(model.zt).item()
        grad_norm = torch.linalg.norm(grad_val.reshape(-1)).item()
        print(f"Epoch {idx}: Loss = {self.reg_energy(model.zt).item():.4f}")
        if self.save_path is not None:
            torch.save({
                        'epoch': idx,
                        'zt': model.zt,
                        'grad_norm': grad_norm,
                        'reg_energy': reg_energy,
                        },
                        f'{self.save_path}georcep_epoch_{idx}.pt'
                        )
        
        return
    
    def __call__(self, 
                 z0:Tensor,
                 zT:Tensor,
                 step:str="while",
                 )->None:
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_xt,
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0.detach()
        self.zT = zT.detach()
        self.diff = zT-z0
        self.dim = len(z0)
        
        zt = self.init_fun(z0,zT,self.T)
        model = GeoCurve(z0, zt, zT)

        energy_init = self.energy(zt).item()
        score_norm2_init = self.score_norm2(zt).item()
        self.lam_norm = self.lam*score_norm2_init/energy_init
        gt = self.gt(model)#jnp.einsum('tj,tjid,ti->td', un[1:], self.M.DG(xn[1:-1]), un[1:])
        grad_val = self.Dregenergy(model.ut(zt), gt)
        print(f"Epoch 0: Loss = {self.reg_energy(model.zt).item():.4f}")
        model.train()
        if step == "while":
            idx = 0
            while self.cond_fun(grad_val, idx):
                model, gt, grad_val, idx = self.georce_step(model, gt, grad_val, idx)
                if idx % self.save_step == 0:
                    self.save_model(model, grad_val, idx)
        elif step == "for":
            for idx in range(self.max_iter):
                model, gt, grad_val, idx = self.georce_step(model, gt, grad_val, idx)
                if idx % self.save_step == 0:
                    self.save_model(model, grad_val, idx)
        else:
            raise ValueError(f"step argument should be either for or while. Passed argument is {step}")
            
            
        self.save_model(model, grad_val, idx)
        reg_energy = self.reg_energy(model.zt).item()
        grad_norm = torch.linalg.norm(grad_val.reshape(-1)).item()
        
        zt = torch.vstack((z0, model.zt, zT)).detach()
            
        return zt, reg_energy, grad_norm, idx