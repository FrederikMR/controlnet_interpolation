#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 12:01:26 2024

@author: fmry
"""

#%% Sources

#%% Modules

import torch
from torch import vmap

from torch import Tensor
from typing import Callable, Dict, Tuple
from abc import ABC

from .utils import GeoCurve
from torch_geometry.manifolds import RiemannianManifold
from torch_geometry.line_search import Backtracking

#%% Gradient Descent Estimation of Geodesics

class ProbGEORCE(ABC):
    """Probabilistic GEORCE

    Estimates geodesics under a Riemannian metric, G, under the constaint that
    these are within the probability distribituon using the score function.

    Attributes:
        M: A Riemannian manifold
        reg_fun: regularizing function that is vectorized and returns a scalar
        init_fun: initilization function for the initial curve
        lam: lambda that determines the deviation between the geodesic and probaility flow
        N: number of grid points, in total the outputet curve will have (N+1) grid points
        tol: the tolerance for convergence
        max_iter: the maximum number of iterations
        line_search_params: the parameters for the line search (max_iter and rho), where rho is backtracking parameter
    """

    def __init__(self,
                 M:RiemannianManifold | None,
                 reg_fun:Callable[[Tensor], float],
                 init_fun:Callable[[Tensor,Tensor,int],Tensor]|None=None,
                 lam:float=1.0,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'max_iter': 100, 'rho': 0.5},
                 clip:bool=False,
                 boundary:float=2.0,
                 device:str=None,
                 )->None:
        """Initializes the instance of ProbGEORCE.

        Args:
          M: A Riemannian manifold
          reg_fun: regularizing function that is vectorized
          init_fun: initilization function for the initial curve
          lam: lambda that determines the deviation between the geodesic and probaility flow
          N: number of grid points, in total the outputet curve will have (N+1) grid points
          tol: the tolerance for convergence
          max_iter: the maximum number of iterations
          line_search_params: the parameters for the line search (max_iter and rho), where rho is backtracking parameter
        """
        
        self.M = M
        self.reg_fun = reg_fun
        
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        self.clip = clip
        self.boundary = boundary
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if init_fun is None:
            self.init_fun = lambda z0, zN, N: (zN-z0)*torch.linspace(0.0,
                                                                     1.0,
                                                                     N+1,
                                                                     dtype=z0.dtype,
                                                                     device=self.device)[1:-1].reshape(-1,1)+z0
        else:   
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return f"Geodesic Computation Object using Control Problem with Probability Flow \
            \n Parameters are: \n-\t lambda={self.lam:.4f}, \n\t-N={self.N}, \n\t-tol={self.tol:.4f} \
                \n\t-max_iter={self.max_iter}"
    
    @torch.no_grad()
    def reg_norm2(self, 
                  zi:Tensor,
                  )->Tensor:
        """Computes the regularizing term for the energy

        Args:
          zi: curve
         Output:
           the regularizing term for the energy
        """
        
        return self.reg_fun(zi)
    
    @torch.no_grad()
    def energy(self, 
               zi:Tensor,
               )->Tensor:
        """Computes the energy of the geodesic

        Args:
          zi: curve
         Output:
           the energy of the geodesic
        """
        
        dz0 = zi[0]-self.z0
        e1 = torch.einsum('i,ij,j->', dz0, self.G0, dz0)
        
        zi = torch.vstack((zi, self.zN))
        Gi = vmap(self.M.G)(zi[:-1])
        dzi = zi[1:]-zi[:-1]
        
        return e1+torch.sum(torch.einsum('...i,...ij,...j->...', dzi, Gi, dzi))

    @torch.no_grad()
    def reg_energy(self, 
                   zi:Tensor,
                   *args,
                   )->Tensor:
        """Computes the regularized energy of the geodesic and regularizing function

        Args:
          zi: curve
         Output:
           the regularized energy of the geodesic and regularizing function
        """
        
        energy = self.energy(zi)
        reg_term = self.reg_norm2(zi)
        
        return energy + self.lam_norm*reg_term
    
    @torch.no_grad()
    def Dregenergy(self,
                   zi:Tensor,
                   ui:Tensor,
                   Gi:Tensor,
                   gi:Tensor,
                   )->Tensor:
        """Computes the gradient of regularized energy of the geodesic and regularizing function

        Args:
          zi: curve
          ui: velocity along the curve
          Gi: metric matrix function
          gi: gradient of the regularized inner product
         Output:
           the gradient of regularized energy of the geodesic and regularizing function
        """
        
        denergy = gi+2.*(torch.einsum('tij,tj->ti', Gi[:-1], ui[:-1])-torch.einsum('tij,tj->ti', Gi[1:], ui[1:]))

        return denergy
    
    def inner_product(self,
                      zi:Tensor,
                      ui:Tensor,
                      )->Tensor:
        """Computes the inner product of regularized energy of the geodesic and regularizing function

        Args:
          zi: curve
          ui: velocity along the curve
         Output:
           the inner product of regularized energy of the geodesic and regularizing function
        """
        
        Gi = vmap(self.M.G)(zi)
        reg_term = self.reg_fun(zi)
        
        return torch.sum(torch.einsum('...i,...ij,...j->...', ui, Gi, ui))+self.lam_norm*reg_term, Gi.detach()
    
    def gi(self,
           model:GeoCurve,
           ui:Tensor,
           )->Tuple[Tensor, Tensor]:
        """Computes the gradient inner product of regularized energy of the geodesic and regularizing functionn

        Args:
          model: geodesic curve
          ui: velocity along the curve
         Output:
           gi: the gradient inner product of regularized energy of the geodesic and regularizing function
           Gi: metric matrix function
        """
        
        zi = model.forward()
        loss, Gi = self.inner_product(zi, ui[1:])
        loss.backward()
        gi = model.zi.grad

        Gi = torch.vstack((self.G0.reshape(-1,self.M.dim,self.M.dim),
                           Gi))
        
        return gi.detach(), Gi.detach()
    
    @torch.no_grad()
    def update_scheme(self, 
                      gi:Tensor, 
                      Gi_inv:Tensor,
                      )->Tensor:
        """Computes dual variables for ProbGEORCE
        
        Args:
          gi: gradient of the regularized inner product
          Gi_inv: inverse of metric matrix function
         Output:
           mui: the dual variables in the control problem
        """

        g_cumsum = torch.flip(torch.cumsum(torch.flip(gi, dims=[0]), dim=0), dims=[0])
        Ginv_sum = torch.sum(Gi_inv, dim=0)
        
        rhs = torch.sum(torch.einsum('tij,tj->ti', Gi_inv[:-1], g_cumsum), dim=0)+2.0*self.diff
        
        muN = -torch.linalg.solve(Ginv_sum, rhs)
        mui = torch.vstack((muN+g_cumsum, muN))
        
        return mui
    
    @torch.no_grad()
    def update_zi(self,
                  zi:Tensor,
                  alpha:Tensor,
                  ui_hat:Tensor,
                  ui:Tensor,
                  )->Tensor:
        """Update curve points
        
        Args:
          zi: discretized curve points
          alpha: step size
          ui_hat: candidate updatre velocity along the curve
          ui: velocity along current curve
         Output:
           updated discretized curve points
        """
        
        return self.z0+torch.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], dim=0)
    
    @torch.no_grad()
    def update_ui(self,
                  Gi_inv:Tensor,
                  mui:Tensor,
                  )->Tensor:
        
        """Computes updated values of the velocity along the curve
        
        Args:
          Gi_inv: inverse of metric matrix function
          mui: dual variables
         Output:
           updated values of the velocity along the curve
        """
        
        return -0.5*torch.einsum('tij,tj->ti', Gi_inv, mui)

    def cond_fun(self, 
                 grad_val:Tensor,
                 idx:int,
                 )->Tensor:
        """Computes boolean value for stopping if converged
        
        Args:
          grad_val: gradient of the regularized energy function
          idx: iteration number
         Output:
           boolean value indicating convergence if gradient is less than tolerance
        """
        
        grad_norm = torch.linalg.norm(grad_val.reshape(-1)).item()

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                    model:GeoCurve,
                    Gi:Tensor,
                    Gi_inv:Tensor,
                    gi:Tensor,
                    grad_val:Tensor,
                    idx:int,
                    )->Tuple[GeoCurve, Tensor, Tensor, Tensor, Tensor, int]:
        """Computes GEORCE iteration
        
        Args:
          model: geodesic curve object
          Gi: metric matrix function
          Gi_inv: inverse metric matrix function
          gi: gradient of the regularized inner product
          grad_val: gradient of the regularized energy function
          idx: number of iterations
         Output:
           model: updated geodesic curve object
           Gi: metric matrix function
           Gi_inv: inverse metric matrix function
           gi: gradient of the regularized inner product
           grad_val: gradient of the regularized energy function
           idx: number of iterations
        """
        
        zi = model.zi
        ui = model.ui(zi)

        mui = self.update_scheme(gi, Gi_inv)
        
        ui_hat = self.update_ui(Gi_inv, mui)
        tau = self.line_search(zi, grad_val, ui_hat, ui)

        ui = tau*ui_hat+(1.-tau)*ui
        with torch.no_grad():
            model.zi = torch.nn.Parameter(self.z0+torch.cumsum(ui[:-1], dim=0))
            
        gi, Gi = self.gi(model,ui)
        Gi_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gi[1:])))    
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
            
        return model, Gi, Gi_inv, gi, grad_val, idx+1
    
    def __call__(self, 
                 z0:Tensor,
                 zN:Tensor,
                 )->Tensor:
        """Interpolation between z0 and zN
        
        Args:
          z0: tensor
          zN: tensor
         Output:
           zi: interpolation curve between z0 and zN
        """
        
        shape = z0.shape
        
        #if self.clip:
        #    z0 = torch.clip(z0, -self.boundary, self.boundary)
        #    zN = torch.clip(zN, -self.boundary, self.boundary)
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_zi,
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0.reshape(-1).detach()
        self.zN = zN.reshape(-1).detach()
        self.diff = self.zN-self.z0
        self.dim = len(self.z0)

        self.G0 = self.M.G(z0).detach()
        self.Ginv0 = torch.linalg.inv(self.G0).reshape(1,self.dim,self.dim)
        
        zi = self.init_fun(self.z0,self.zN,self.N)
        model = GeoCurve(self.z0, zi, self.zN)
        ui = model.ui(zi)

        energy_init = self.energy(zi).item()
        reg_term_init = self.reg_norm2(zi).item()
        self.lam_norm = self.lam*energy_init/reg_term_init
        
        gi, Gi = self.gi(model,ui)
        Gi_inv = torch.vstack((self.Ginv0, torch.linalg.inv(Gi[1:])))    
        grad_val = self.Dregenergy(zi, ui, Gi, gi)
        model.train()
        
        idx = 0
        grad_val = torch.ones_like(zi)+self.tol
        while self.cond_fun(grad_val, idx):
            model, Gi, Gi_inv, gi, grad_val, idx = self.georce_step(model, 
                                                                    Gi,
                                                                    Gi_inv,
                                                                    gi,
                                                                    grad_val, 
                                                                    idx,
                                                                    )

        zi = torch.vstack((self.z0, model.zi, self.zN)).detach()
        
        if self.clip:
            zi = torch.clip(zi, -self.boundary, self.boundary)
            
        return zi.reshape(-1,*shape)

#%% Probabilistic GEORCE for Euclidean Background Metric

class ProbEuclideanGEORCE(ABC):
    """Probabilistic GEORCE with Euclidean background metric

    Estimates geodesics for the Euclidean metric under the constaint that
    these are within the probability distribituon using the score function.

    Attributes:
        reg_fun: regularizing function that is vectorized and returns a scalar
        init_fun: initilization function for the initial curve
        lam: lambda that determines the deviation between the geodesic and probaility flow
        N: number of grid points, in total the outputet curve will have (N+1) grid points
        tol: the tolerance for convergence
        max_iter: the maximum number of iterations
        line_search_params: the parameters for the line search (max_iter and rho), where rho is backtracking parameter
    """
    def __init__(self,
                 reg_fun:Callable,
                 init_fun:Callable=None,
                 lam:float=1.0,
                 N:int=100,
                 tol:float=1e-4,
                 max_iter:int=1000,
                 line_search_params:Dict = {'rho': 0.5},
                 device:str=None,
                 clip:bool=True,
                 boundary:float=2.0,
                 )->None:
        """Initializes the instance of ProbGEORCE with Euclidean background metric.

        Args:
          reg_fun: regularizing function that is vectorized
          init_fun: initilization function for the initial curve
          lam: lambda that determines the deviation between the geodesic and probaility flow
          N: number of grid points, in total the outputet curve will have (N+1) grid points
          tol: the tolerance for convergence
          max_iter: the maximum number of iterations
          line_search_params: the parameters for the line search (max_iter and rho), where rho is backtracking parameter
        """

        self.reg_fun = reg_fun
        
        self.lam = lam
        self.N = N
        self.tol = tol
        self.max_iter = max_iter
        self.line_search_params = line_search_params
        
        self.clip = clip
        self.boundary = boundary
        
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
        if init_fun is None:
            self.init_fun = lambda z0, zN, N: (zN-z0)*torch.linspace(0.0,
                                                                     1.0,
                                                                     N+1,
                                                                     dtype=z0.dtype,
                                                                     device=self.device)[1:-1].reshape(-1,1)+z0
        else:
            self.init_fun = init_fun
        
    def __str__(self)->str:
        
        return f"Geodesic Computation Object using Control Problem with Probability Flow for Euclidean metric \
            \n Parameters are: \n-\t lambda={self.lam:.4f}, \n\t-N={self.N}, \n\t-tol={self.tol:.4f} \
                \n\t-max_iter={self.max_iter}"

    @torch.no_grad()
    def reg_norm2(self, 
                    zi:Tensor,
                    )->Tensor:
        """Computes the regularizing term for the energy

        Args:
          zi: curve
         Output:
           the regularizing term for the energy
        """
        
        return self.reg_fun(zi)

    @torch.no_grad()
    def energy(self, 
               zi:Tensor,
               )->Tensor:
        """Computes the energy of the geodesic

        Args:
          zi: curve
         Output:
           the energy of the geodesic
        """

        zi = torch.vstack((self.z0, zi, self.zN))
        ui = zi[1:]-zi[:-1]
        
        return torch.sum(ui*ui)
    
    @torch.no_grad()
    def reg_energy(self, 
                   zi:Tensor,
                   *args,
                   )->Tensor:
        """Computes the regularized energy of the geodesic and regularizing function

        Args:
          zi: curve
         Output:
           the regularized energy of the geodesic and regularizing function
        """

        score_norm2 = self.reg_norm2(zi)
        energy = self.energy(zi)
        
        return energy+self.lam_norm*score_norm2
    
    @torch.no_grad()
    def Dregenergy(self,
                   ui:Tensor,
                   gi:Tensor,
                   )->Tensor:
        """Computes the gradient of regularized energy of the geodesic and regularizing function

        Args:
          zi: curve
          ui: velocity along the curve
         Output:
           the gradient of regularized energy of the geodesic and regularizing function
        """
        
        return (gi+2.*(ui[:-1]-ui[1:])).reshape(-1)

    def inner_product(self,
                      zi:Tensor,
                      )->Tensor:
        """Computes the inner product of regularized energy of the geodesic and regularizing function

        Args:
          zi: curve
         Output:
           the inner product of regularized energy of the geodesic and regularizing function
        """

        return self.lam_norm*self.reg_fun(zi)
    
    def gi(self,
           model:GeoCurve,
           )->Tensor:
        """Computes the gradient inner product of regularized energy of the geodesic and regularizing function

        Args:
          model: geodesic curve
         Output:
           gi: the gradient inner product of regularized energy of the geodesic and regularizing function
        """
        
        zi = model.forward()
        loss = self.inner_product(zi)
        loss.backward()
        gi = model.zi.grad

        return gi.detach()
        
    @torch.no_grad()
    def update_zi(self,
                  zi:Tensor,
                  alpha:Tensor,
                  ui_hat:Tensor,
                  ui:Tensor,
                  )->Tensor:
        """Update curve points
        
        Args:
          zi: discretized curve points
          alpha: step size
          ui_hat: candidate updatre velocity along the curve
          ui: velocity along current curve
         Output:
           updated discretized curve points
        """
        
        return self.z0+torch.cumsum(alpha*ui_hat[:-1]+(1.-alpha)*ui[:-1], dim=0)
    
    @torch.no_grad()
    def update_ui(self,
                  gi:Tensor,
                  )->Tensor:
        """Computes updated values of the velocity along the curve
        
        Args:
          gi: the gradient inner product of regularized energy of the geodesic and regularizing function
         Output:
           updated values of the velocity along the curve
        """
        
        g_cumsum = torch.vstack((torch.flip(torch.cumsum(torch.flip(gi, dims=[0]), dim=0), dims=[0]), 
                                 torch.zeros(self.dim, device=self.device)))
        g_sum = torch.sum(g_cumsum, dim=0)/self.N
        
        return self.diff/self.N+0.5*(g_sum-g_cumsum)

    def cond_fun(self, 
                 grad_val:Tensor,
                 idx:int,
                 )->Tensor:
        """Computes boolean value for stopping if converged
        
        Args:
          grad_val: gradient of the regularized energy function
          idx: iteration number
         Output:
           boolean value indicating convergence if gradient is less than tolerance
        """
        
        grad_norm = torch.linalg.norm(grad_val.reshape(-1))

        return (grad_norm>self.tol) & (idx < self.max_iter)
    
    def georce_step(self,
                   model:GeoCurve,
                   gi:Tensor,
                   grad_val:Tensor,
                   idx:int,
                   )->Tensor:
        """Computes GEORCE iteration
        
        Args:
          model: geodesic curve object
          gi: gradient of the regularized inner product
          grad_val: gradient of the regularized energy function
          idx: number of iterations
         Output:
           model: updated geodesic curve object
           gi: gradient of the regularized inner product
           grad_val: gradient of the regularized energy function
           idx: number of iterations
        """
        
        zi = model.zi
        ui = model.ui(zi)
        
        ui_hat = self.update_ui(gi)
        tau = self.line_search(zi, grad_val, ui_hat, ui)

        ui = tau*ui_hat+(1.-tau)*ui
        with torch.no_grad():
            model.zi = torch.nn.Parameter(self.z0+torch.cumsum(ui[:-1], dim=0))
        
        gi = self.gi(model)
        grad_val = self.Dregenergy(ui, gi)
        
        return model, gi, grad_val, idx+1
    
    def __call__(self, 
                 z0:Tensor,
                 zN:Tensor,
                 )->None:
        """Interpolation between z0 and zN
        
        Args:
          z0: tensor
          zN: tensor
         Output:
           zi: interpolation curve between z0 and zN
        """
        
        shape = z0.shape
        
        #if self.clip:
        #    z0 = torch.clip(z0, -self.boundary, self.boundary)
        #    zN = torch.clip(zN, -self.boundary, self.boundary)
        
        self.line_search = Backtracking(obj_fun=self.reg_energy,
                                        update_fun=self.update_zi,
                                        **self.line_search_params,
                                        )
        
        self.z0 = z0.reshape(-1).detach()
        self.zN = zN.reshape(-1).detach()
        self.diff = self.zN-self.z0
        self.dim = len(self.z0)
        
        zi = self.init_fun(self.z0,self.zN,self.N)
        model = GeoCurve(self.z0, zi, self.zN)

        energy_init = self.energy(zi).item()
        reg_norm2_init = self.reg_norm2(zi).item()
        self.lam_norm = self.lam*energy_init/reg_norm2_init
        gi = self.gi(model)
        grad_val = self.Dregenergy(model.ui(zi), gi)
        model.train()
        
        idx = 0
        while self.cond_fun(grad_val, idx):
            model, gi, grad_val, idx = self.georce_step(model, gi, grad_val, idx)
        
        zi = torch.vstack((self.z0, model.zi, self.zN)).detach()
        
        if self.clip:
            zi = torch.clip(zi, -self.boundary, self.boundary)
            
        return zi.reshape(-1, *shape)
    
#%% ProbGEORCE NoiseDiffusion

class ProbGEORCE_ND(ABC):
    """Probabilistic GEORCE with NoiseDiffusion

    Estimates geodesics for the Euclidean metric under the constaint that
    these are within the probability distribituon using the score function.

    Attributes:
        reg_fun: regularizing function that is vectorized and returns a scalar
        init_fun: initilization function for the initial curve
        lam: lambda that determines the deviation between the geodesic and probaility flow
        N: number of grid points, in total the outputet curve will have (N+1) grid points
        tol: the tolerance for convergence
        max_iter: the maximum number of iterations
        line_search_params: the parameters for the line search (max_iter and rho), where rho is backtracking parameter
    """
    def __init__(self,
                 interpolater:Callable[[torch.Tensor, torch.Tensor], torch.Tensor],
                 alpha:Callable=lambda s: torch.cos(0.5*torch.pi*s),
                 beta:Callable=lambda s: torch.sin(0.5*torch.pi*s),
                 mu:Callable|None= lambda s: None,
                 nu:Callable|None= lambda s: None,
                 gamma:float=0.0,
                 sigma:float=1.0,
                 boundary:float=2.0,
                 device:str=None,
                 )->None:
        """Initializes the instance of ProbGEORCE with Euclidean background metric.

        Args:
          reg_fun: regularizing function that is vectorized
          init_fun: initilization function for the initial curve
          lam: lambda that determines the deviation between the geodesic and probaility flow
          N: number of grid points, in total the outputet curve will have (N+1) grid points
          tol: the tolerance for convergence
          max_iter: the maximum number of iterations
          line_search_params: the parameters for the line search (max_iter and rho), where rho is backtracking parameter
        """
        
        self.interpolater = interpolater
        
        self.alpha = alpha
        self.beta = beta

        self.mu = mu if mu is None else lambda s: 1.2*self.alpha(s)/(self.alpha(s)+self.beta(s))
        self.nu = nu if nu is None else lambda s: 1.2*self.beta(s)/(self.alpha(s)+self.beta(s))
        
        self.gamma = gamma
        self.sigma = sigma
        self.boundary = boundary
        if device is None:
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        else:
            self.device = device
        
    def __str__(self)->str:
        
        return "NoiseDiffusion with different ProbGEORCE interpolation"

    def __call__(self,
                 z0:Tensor,
                 zN:Tensor,
                 x0:Tensor,
                 xN:Tensor,
                 )->Tensor:
        
        shape = z0.shape
        
        z0 = z0.reshape(-1)
        zN = zN.reshape(-1)
        x0 = x0.reshape(-1)
        xN = xN.reshape(-1)
        
        s = torch.linspace(0,1,self.interpolater.N+1,
                           device=self.device,
                           )[1:-1].reshape(-1,1)

        #alpha=math.cos(math.radians(s*90))
        #beta=math.sin(math.radians(s*90))
        alpha = torch.cos(0.5*torch.pi*s)
        beta = torch.sin(0.5*torch.pi*s)
        
        mu = vmap(self.mu)(s)
        nu = vmap(self.nu)(s)
        eps = self.sigma*torch.randn_like(z0)
        
        l=alpha/beta
        
        alpha=((1-self.gamma*self.gamma)*l*l/(l*l+1))**0.5
        beta=((1-self.gamma*self.gamma)/(l*l+1))**0.5
        
        noise_curve = self.interpolater(z0,zN)[1:-1]
        data_curve = self.interpolater(x0, xN)[1:-1]
        
        noise_latent = noise_curve - data_curve + \
            (mu*x0 + nu * xN)+self.gamma*eps

        curve=torch.clip(noise_latent,-self.boundary,self.boundary)
        curve = torch.vstack((z0, curve, zN))
        
        return curve.reshape(-1, *shape)
