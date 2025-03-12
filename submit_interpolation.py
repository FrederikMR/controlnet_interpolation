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
    
    os.system("bsub < submit_interpolation.sh")
    
    return

#%% Generate jobs

def generate_job(model, method, lam, clip, N, max_iter=100):

    with open ('submit_interpolation.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J {model}_{method}
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
        --N {N} \\
        --max_iter {max_iter} \\
        --ckpt_path /work3/fmry/models/controlnet/control_v11p_sd21_openpose.ckpt \\
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    N = 10
    max_iter = 100
    model = ['aircraft']
    method = ['ProbGEORCE_DND', 'ProbGEORCE_D', 'linear', 'noisediffusion', 'slerp', 'noise']
    clip = [0]
    lam = [0.1, 0.5, 1.0, 10.0]
    
    for mod in model:
        for meth in method:
            for cl in clip:
                time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
                if "ProbGEORCE" in meth:
                    for l in lam:
                        generate_job(model=mod, method=meth, lam=l, clip=cl, N=N, max_iter=max_iter)
                        try:
                            submit_job()
                        except:
                            time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                            try:
                                submit_job()
                            except:
                                print(f"Job script with {mod}, {meth} failed!")
                else:
                    generate_job(model=mod, method=meth, lam=1.0, clip=cl, N=N, max_iter=max_iter)
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
