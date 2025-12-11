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
import os

import cm
from load_data import load_dataset

#%% Args Parser

def parse_args():
    parser = argparse.ArgumentParser()
    # File-paths
    parser.add_argument('--img_types', default="bedroom",
                        type=str)
    parser.add_argument('--method', default="ProbGEORCE",
                        type=str)
    parser.add_argument('--computation_method', default="ivp", #bvp, ivp, mean
                        type=str)
    parser.add_argument('--n_images', default=10,
                        type=int)
    parser.add_argument('--image_size', default=768,
                        type=int)
    parser.add_argument('--target_prompt', default=1,
                        type=int)
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
    parser.add_argument('--num_images', default=10,
                        type=int)
    parser.add_argument('--seed', default=2712,
                        type=int)
    parser.add_argument('--ckpt_path', default="models/control_v11p_sd21_openpose.ckpt",
                        type=str)

    args = parser.parse_args()
    return args

#%% Run interpolation

def run_interpolation()->None:
    
    args = parse_args()
    
    imgs, prompt, target_prompt, n_prompt = load_dataset(name = args.img_types,
                                                         n_images=args.n_images, 
                                                         image_size=args.image_size,
                                                         )
    
    #osp = os.path
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    #sys.path.append(osp.expandvars('$NFS/NoiseDiffusion/controlnet'))
    
    if args.mu < 0:
        mu = None
    else:
        mu = args.mu
        
    if args.nu < 0:
        nu = None
    else:
        nu = args.nu
    
    CM = cm.ContextManager(N=args.N, lam=args.lam, max_iter=args.max_iter, inter_method=args.method, clip=args.clip,
                           mu = mu, nu = nu,
                           ckpt_path=args.ckpt_path, num_images=args.num_images, seed=args.seed)
    
    if args.computation_method == "ivp":
        if args.target_prompt:
            CM.ivp(imgs[0], prompt_neutral=prompt, prompt_target = target_prompt, n_prompt=n_prompt, ddim_steps=200,  guide_scale=10,  out_dir=f'../figures/{args.img_types}/')
        else:
            CM.ivp(imgs[0], prompt_neutral=prompt, prompt_target = prompt, n_prompt=n_prompt, ddim_steps=200,  guide_scale=10,  out_dir=f'../figures/{args.img_types}/')
    elif args.computation_method == "bvp":
        CM.bvp(imgs[0], imgs[1], prompt=prompt, n_prompt=n_prompt, ddim_steps=200,  guide_scale=10,  out_dir=f'../figures/{args.img_types}/')
    elif args.computation_method == "mean":
        CM.mean(imgs, prompt=prompt, n_prompt=n_prompt, ddim_steps=200,  guide_scale=10,  out_dir=f'../figures/{args.img_types}/')
        
    return

if __name__ == '__main__':
    
    run_interpolation()



