#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar  9 03:05:05 2025

@author: fmry
"""

#%% Sources

#Checkpoints: https://huggingface.co/thibaud/controlnet-sd21/tree/main

#%% Modules

import torch

#argparse
import argparse

from sample_scripts.cat import run_cat_timing

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--model', default="bedroom",
                        type=str)
    parser.add_argument('--method', default="ProbGEORCE",
                        type=str)
    parser.add_argument('--lam', default=1.0,
                        type=float)
    parser.add_argument('--clip', default=0,
                        type=int)
    parser.add_argument('--N', default=10,
                        type=int)
    parser.add_argument('--mu', default=-1.,
                        type=float)
    parser.add_argument('--nu', default=-1.,
                        type=float)
    parser.add_argument('--max_iter', default=100,
                        type=int)
    parser.add_argument('--ckpt_path', default="models/control_v11p_sd21_openpose.ckpt",
                        type=str)

    args = parser.parse_args()
    return args

#%% Run interpolation

def run_interpolation()->None:
    
    args = parse_args()
    
    run_fun = run_cat_timing
        
    if args.mu < 0:
        mu = None
    else:
        mu = args.mu
        
    if args.nu < 0:
        nu = None
    else:
        nu = args.nu

    run_fun(N = args.N, 
            lam = args.lam, 
            max_iter = args.max_iter, 
            inter_method = args.method,
            mu = mu,
            nu = nu,
            clip=args.clip,
            ckpt_path=args.ckpt_path,
            )

    return

if __name__ == '__main__':
    
    run_interpolation()



