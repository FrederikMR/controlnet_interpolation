#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 15 17:22:08 2024

@author: fmry
"""

#%% Modules

import numpy as np

import os

import time

#%% Submit job

def submit_job():
    
    os.system("bsub < submit_modification.sh")
    
    return

#%% Generate jobs

def generate_job(model, method, lam, clip, N, mu, nu, max_iter=100):

    with open ('submit_modification.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J {model}_{method}_{mu}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=16GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o sendmeemail/error_%J.out 
    #BSUB -e sendmeemail/output_%J.err 
    
    module swap python3/3.10.12
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    
    python3 run_interpolation.py \\
        --model {model} \\
        --method {method} \\
        --lam {lam} \\
        --clip {clip} \\
        --mu {mu} \\
        --nu {nu} \\
        --N {N} \\
        --max_iter {max_iter} \\
        --ckpt_path /work3/fmry/models/controlnet/control_v11p_sd21_openpose.ckpt \\
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    N = 10
    max_iter = 100
    model = ['eagle', 'president']
    method = ['ProbGEORCE_ND', 'NoiseDiffusion']
    clip = [1]
    lam = [1.0]
    
    modifier = [0.5, 0.625, 0.750, 0.875, 1.0]
    
    for mod in model:
        for meth in method:
            for cl in clip:
                for mu in modifier:
                    time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
                    if "ProbGEORCE" in meth:
                        for l in lam:
                            generate_job(model=mod, method=meth, lam=l, clip=cl, N=N, mu=mu, nu=mu, max_iter=max_iter)
                            try:
                                submit_job()
                            except:
                                time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                                try:
                                    submit_job()
                                except:
                                    print(f"Job script with {mod}, {meth} failed!")
                    else:
                        generate_job(model=mod, method=meth, lam=1.0, clip=cl, N=N, mu=mu, nu=mu, max_iter=max_iter)
                        try:
                            submit_job()
                        except:
                            time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                            try:
                                submit_job()
                            except:
                                print(f"Job script with {mod}, {meth} failed!")


#%% main

if __name__ == '__main__':
    
    loop_jobs(1.0)
