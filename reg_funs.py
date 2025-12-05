#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  5 15:57:07 2025

@author: fmry
"""

#%% Modules

import torch

from torch.distributions.chi2 import Chi2
from torch.distributions.normal import Normal

from abc import ABC


#%% Reg funs

class RegFuns(ABC):
    def __init__(self,
                 model,
                 dim:int,
                 )->None:
        
        self.model = model
        self.chi2 = Chi2(dim)
        self.normal = Normal(loc=0.0, scale=dim**0.5)
        
        return
    
    def __str__(self,)->str:
        
        return "Regularization functions"
    
    def chi2_reg(self, x:torch.Tensor)->torch.Tensor:
        
        return -torch.sum(self.chi2.log_prob(torch.sum(x**2, axis=-1)))
    
    def sphere_reg(self, x:torch.Tensor)->torch.Tensor:
        
        return torch.sum((torch.sum(x**2, axis=1)-self.dim)**2)
    
    def ip_reg(self, x:torch.Tensor)->torch.Tensor:
        
        """
        Compute log-probabilities of pairwise inner products
        under N(0, d) approximation, where d = x.shape[1].
        
        Args:
            x: Tensor of shape (N, d)
            return_matrix: If True, return full log-prob matrix with diag masked.
                           Otherwise return the sum of all log-probs.
        """
        N, d = x.shape
    
        # Pairwise inner products
        G = x @ x.t()                              # (N, N)
    
        # Remove diagonal (self-inner products)
        mask = ~torch.eye(N, dtype=torch.bool, device=x.device)
        ip = G[mask]                                # (N*(N-1),)
    
        # Log probability of each inner product
        logp = self.normal.log_prob(ip)
    
        # Otherwise return total log probability
        return -logp.sum()
    
    def diff_reg(self, x:torch.Tensor)->torch.Tensor:
        
        """
        Compute sum of log-probabilities of all pairwise squared distances
        ||X_i - X_j||^2 under 2 * chi^2_d, optimized for large N and high d.
    
        Args:
            X: tensor of shape (N, d)
        Returns:
            scalar: sum of log-probabilities over all i < j
        """
        N, d = x.shape
        device = x.device
    
        # Compute squared pairwise distances efficiently
        # ||Xi - Xj||^2 = ||Xi||^2 + ||Xj||^2 - 2 Xi·Xj
        XX = x @ x.t()                   # (N, N)
        diag = torch.sum(x**2, dim=1)    # (N,)
        dist2 = diag[:, None] + diag[None, :] - 2 * XX  # (N, N)
    
        # Mask upper triangle (i < j)
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        dist2_pairs = dist2[mask]  # shape (N*(N-1)/2,)
    
        # Log PDF of 2 * chi^2_d
        nu = d / 2
        log_pdf = -nu * torch.log(torch.tensor(2.0, device=device)) \
                  - torch.lgamma(torch.tensor(nu, device=device)) \
                  + (nu - 1) * torch.log(dist2_pairs + 1e-20) \
                  - dist2_pairs / 2
    
        # Sum over all pairs
        return -log_pdf.sum()

    def reg_fun3_logprob(X):
        """
        Compute sum of log-probabilities of all pairwise squared distances
        ||X_i - X_j||^2 under 2 * chi^2_d, optimized for large N and high d.
    
        Args:
            X: tensor of shape (N, d)
        Returns:
            scalar: sum of log-probabilities over all i < j
        """
        N, d = X.shape
        device = X.device
    
        # Compute squared pairwise distances efficiently
        # ||Xi - Xj||^2 = ||Xi||^2 + ||Xj||^2 - 2 Xi·Xj
        XX = X @ X.t()                   # (N, N)
        diag = torch.sum(X**2, dim=1)    # (N,)
        dist2 = diag[:, None] + diag[None, :] - 2 * XX  # (N, N)
    
        # Mask upper triangle (i < j)
        mask = torch.triu(torch.ones(N, N, device=device, dtype=torch.bool), diagonal=1)
        dist2_pairs = dist2[mask]  # shape (N*(N-1)/2,)
    
        # Log PDF of 2 * chi^2_d
        nu = d / 2
        log_pdf = -nu * torch.log(torch.tensor(2.0, device=device)) \
                  - torch.lgamma(torch.tensor(nu, device=device)) \
                  + (nu - 1) * torch.log(dist2_pairs + 1e-20) \
                  - dist2_pairs / 2
    
        # Sum over all pairs
        return log_pdf.sum()



    
    def reg_fun3(X):
        """
        X: tensor of shape (N, d)
        Returns: scalar — sum over i<j of (||Xi - Xj||^2 - 2d)^2
        """
        N, d = X.shape
    
        # Compute squared pairwise distances (N×N)
        # ||Xi – Xj||² = Xi·Xi + Xj·Xj – 2 Xi·Xj
        XX = X @ X.t()
        diag = torch.diag(XX)
        dist2 = diag[:, None] + diag[None, :] - 2 * XX
    
        # Extract upper triangle (i < j), exclude diagonal
        i, j = torch.triu_indices(N, N, offset=1)
        dist2_pairs = dist2[i, j]
    
        # Compute error (||Xi - Xj||² - 2d)² and sum
        error = (dist2_pairs - 2 * d).pow(2).sum()
    
        return error
    
    def __call__(self, z:torch.Tensor):
        
        return