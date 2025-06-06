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

from sample_scripts.aircraft import run_aircraft
from sample_scripts.apple import run_apple
from sample_scripts.banana import run_banana
from sample_scripts.bedroom import run_bedroom
from sample_scripts.bee import run_bee
from sample_scripts.bird import run_bird
from sample_scripts.car import run_car
from sample_scripts.cat import run_cat
from sample_scripts.cherry import run_cherry
from sample_scripts.cup import run_cup
from sample_scripts.eagle import run_eagle
from sample_scripts.face import run_face
from sample_scripts.flower import run_flower
from sample_scripts.forest import run_forest
from sample_scripts.grape import run_grape
from sample_scripts.horse import run_horse
from sample_scripts.house import run_house
from sample_scripts.lion_tiger import run_lion_tiger
from sample_scripts.moutain import run_moutain
from sample_scripts.panda import run_panda
from sample_scripts.peach import run_peach
from sample_scripts.pumpkin import run_pumpkin
from sample_scripts.shoes import run_shoes
from sample_scripts.spider import run_spider
from sample_scripts.tree import run_tree
from sample_scripts.run_funny import run_funny
from sample_scripts.run_president import run_president
from sample_scripts.run_canada import run_canada
from sample_scripts.run_australia import run_australia
from sample_scripts.football import run_football

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
    
    if args.model == "aircraft":
        run_fun = run_aircraft
    elif args.model == "apple":
        run_fun = run_apple
    elif args.model == "banana":
        run_fun = run_banana
    elif args.model == "bedroom":
        run_fun = run_bedroom
    elif args.model == "bee":
        run_fun = run_bee
    elif args.model == "bird":
        run_fun = run_bird
    elif args.model == "car":
        run_fun = run_car
    elif args.model == "cat":
        run_fun = run_cat
    elif args.model == "cherry":
        run_fun = run_cherry
    elif args.model == "cup":
        run_fun = run_cup
    elif args.model == "eagle":
        run_fun = run_eagle
    elif args.model == "face":
        run_fun = run_face
    elif args.model == "flower":
        run_fun = run_flower
    elif args.model == "forest":
        run_fun = run_forest
    elif args.model == "grape":
        run_fun = run_grape
    elif args.model == "horse":
        run_fun = run_horse
    elif args.model == "house":
        run_fun = run_house
    elif args.model == "lion_tiger":
        run_fun = run_lion_tiger
    elif args.model == "moutain":
        run_fun = run_moutain
    elif args.model == "panda":
        run_fun = run_panda
    elif args.model == "peach":
        run_fun = run_peach
    elif args.model == "pumpkin":
        run_fun = run_pumpkin
    elif args.model == "shoes":
        run_fun = run_shoes
    elif args.model == "spider":
        run_fun = run_spider
    elif args.model == "tree":
        run_fun = run_tree
    elif args.model == "funny":
        run_fun = run_funny
    elif args.model == "president":
        run_fun = run_president
    elif args.model == "canada":
        run_fun = run_canada
    elif args.model == "australia":
        run_fun = run_australia
    elif args.model == "football":
        run_fun = run_football
        
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



