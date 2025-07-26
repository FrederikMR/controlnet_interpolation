#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri May 24 09:54:30 2024

@author: fmry
"""

#%% Sources

#%% Modules

from jax_geometry.setup import *

#%% Riemannian Manifold

class RiemannianManifold(ABC):
    def __init__(self,
                 G:Callable[[Array], Array]=None,
                 f:Callable[[Array], Array]=None,
                 invf:Callable[[Array],Array]=None,
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
           z:Array
           )->Array:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            return jacfwd(self.f)(z)
        
    def pull_back_metric(self,
                         z:Array
                         )->Array:
        
        if self.f is None:
            raise ValueError("Both the pull-back map is not defined")
        else:
            Jf = self.Jf(z)
            return jnp.einsum('ik,il->kl', Jf, Jf)
    
    def DG(self,
           z:Array
           )->Array:

        return jacfwd(self.G)(z)
    
    def Ginv(self,
             z:Array
             )->Array:
        
        return jnp.linalg.inv(self.G(z))
    
    def christoffel_symbols(self,
                            z:Array
                            )->Array:
        
        Dgx = self.DG(z)
        gsharpx = self.Ginv(z)
        
        return 0.5*(jnp.einsum('im,kml->ikl',gsharpx,Dgx)
                   +jnp.einsum('im,lmk->ikl',gsharpx,Dgx)
                   -jnp.einsum('im,klm->ikl',gsharpx,Dgx))
    
    def geodesic_equation(self,
                          z:Array,
                          v:Array
                          )->Array:
        
        Gamma = self.Chris(z)

        dx1t = v
        dx2t = -jnp.einsum('ikl,k,l->i',Gamma,v,v)
        
        return jnp.hstack((dx1t,dx2t))
    
    def energy(self, 
               gamma:Array,
               )->Array:
        
        T = len(gamma)
        energy, _ = lax.scan(self.energy_path,
                             init=0.0,
                             xs=(gamma[:-1], gamma[1:]-gamma[:-1]),
                             )
        
        return T*energy
    
    def energy_path(self,
                    energy,
                    y:Tuple[Array],
                    )->Array:

        z, dz = y
        
        SG = self.G(z)

        energy += jnp.einsum('i,ij,j->', dz, SG, dz)
        
        return (energy,)*2
    
    def length(self, 
               gamma:Array,
               )->Array:
        
        length, _ = lax.scan(self.length_path,
                             init=0.0,
                             xs=(gamma[:-1], gamma[1:]-gamma[:-1]),
                             )
        
        return length
    
    def length_path(self,
                    length,
                    y:Tuple[Array],
                    )->Array:

        z, dz = y
        
        SG = self.G(z)

        length += jnp.sqrt(jnp.einsum('i,ij,j->', dz, SG, dz))
        
        return (length,)*2
    
#%% Indicator Manifold
    
class IndicatorManifold(ABC):
    def __init__(self,
                 M:RiemannianManifold,
                 extrinsic_batch_size:int=None,
                 intrinsic_batch_size:int=None,
                 seed:int = 2712,
                 )->None:
        
        self.M = M
        if extrinsic_batch_size is None:
            self.extrinsic_batch_size = M.emb_dim
        else:
            self.extrinsic_batch_size = extrinsic_batch_size
        
        if intrinsic_batch_size is None:
            self.intrinsic_batch_size = M.dim
        else:
            self.intrinsic_batch_size = intrinsic_batch_size
            
        self.seed = seed
        self.key = jrandom.key(self.seed)
        
        if self.intrinsic_batch_size is None:
            self.scaling = self.M.emb_dim/extrinsic_batch_size
        else:
            self.scaling = (self.M.emb_dim/self.extrinsic_batch_size)*(self.M.dim/self.intrinsic_batch_size)
        self.SG = self.SG_pull_back
        
        self.extrinsic_batch = jnp.arange(0,self.M.emb_dim,1)
        self.instrinsic_batch = jnp.arange(0,self.M.dim,1)
            
        return
        
    def __str__(self)->str:
        
        return "Riemannian Manifold base object"
    
    def random_batch(self,
                     )->Array:
        
        self.key, subkey = jrandom.split(self.key)

        extrinsic_batch = jrandom.choice(subkey, 
                                         a=self.extrinsic_batch,
                                         shape=(self.extrinsic_batch_size,), 
                                         replace=False,
                                         )

        self.key, subkey = jrandom.split(self.key)

        intrinsic_batch = jrandom.choice(subkey, 
                                         a=self.instrinsic_batch,
                                         shape=(self.M.dim-self.intrinsic_batch_size,), 
                                         replace=False,
                                         )
            
        return extrinsic_batch, intrinsic_batch
    
    def Sf(self, 
           z:Array, 
           extrinsic_batch:Array,
           )->Array:
        
        return self.M.f(z)[extrinsic_batch]
            
    
    def SJ(self, 
           z:Array, 
           extrinsic_batch:Array,
           intrinsic_batch:Array=None,
           )->Array:

        z = z.at[intrinsic_batch].set(lax.stop_gradient(z[intrinsic_batch]))
        
        return jacfwd(self.Sf, argnums=0)(z, extrinsic_batch)
    
    def SG_pull_back(self, 
                     z:Array, 
                     extrinsic_batch:Array,
                     intrinsic_batch:Array,
                     )->Array:
        
        SJf = self.SJ(z, extrinsic_batch, intrinsic_batch)
        
        return self.scaling*jnp.einsum('ik,il->kl', SJf, SJf)
    