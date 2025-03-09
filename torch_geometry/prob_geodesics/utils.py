#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 21:17:01 2025

@author: fmry
"""

#%% Modules

import torch
from torch import Tensor

#%% Geodesic Optimization Module

class GeoCurve(torch.nn.Module):
    """GeoCurve contains parameters for a discretized curved as parameters in a neural network

    Attributes:
        z0: start point of the curve
        zN: end point of the curve
        zi: curve connecting z0 and zN
    """
    def __init__(self, 
                 z0:Tensor,
                 zi:Tensor,
                 zN:Tensor,
                 )->None:
        """Initializes the instance of GeoCurve.

        Args:
          z0: start point of the curve
          zN: end point of the curve
          zi: curve connecting z0 and zN
        """
        super(GeoCurve, self).__init__()
        
        self.z0 = z0
        self.zN = zN
        self.zi = torch.nn.Parameter(zi, requires_grad=True)
        
        return
    
    def ui(self,
           zi:Tensor,
           )->Tensor:
        """Computes velocity along the curve

        Args:
          zi: curve
        Output:
          velocity along the curve
        """
        
        zi = torch.vstack((self.z0, zi, self.zN))
        ui = zi[1:]-zi[:-1]
        
        return ui
    
    def forward(self, 
                )->Tensor:
        """Outputs discretized curve

        Output:
          discretized curve
        """
        
        return self.zi