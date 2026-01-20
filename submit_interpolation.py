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

def generate_job(model, computation_method, method, lam, clip, N, reg_type="score", interpolation_space="noise", max_iter=100):

    with open ('submit_interpolation.sh', 'w') as rsh:
        rsh.write(f'''\
    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J {model}_{method}_{computation_method}
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=16GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o ../output_folder/output_%J.out 
    #BSUB -e ../error_folder/error_%J.err 
    
    module swap python3/3.10.12
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    
    python3 run_interpolation.py \\
        --img_types {model} \\
        --computation_method {computation_method} \\
        --n_images 10 \\
        --image_size 768 \\
        --target_prompt 1 \\
        --method {method} \\
        --lam {lam} \\
        --clip {clip} \\
        --mu -1.0 \\
        --nu -1.0 \\
        --N {N} \\
        --max_iter {max_iter} \\
        --num_images 10 \\
        --reg_type {reg_type} \\
        --interpolation_space {interpolation_space} \\
        --ckpt_path /work3/fmry/models/controlnet/control_v11p_sd21_openpose.ckpt \\
    ''')
    
    return

#%% Loop jobs

def loop_jobs(wait_time = 1.0):
    
    #N = 100
    #max_iter = 100
    #model = ['cat', 'bedroom', 'eagle']#, 'president', 'football']
    #method = ['ProbGEORCE']#, 'Linear', 'NoiseDiffusion', 'Spherical', 'Noise']
    #clip = [1]#[0,1]
    #lam = [1.0]#[0.1, 0.5, 1.0, 10.0]
    
    #Project score to sphere TM
    
    N = 10
    max_iter = 1000
    method = ['ProbGEORCE']
    reg_types = ['score', 'score_naive', 'prior']
    interpolation_space=['noise']
    clip = [0]#[0,1]
    lam = [1.0, 10.0]#[0.1, 0.5, 1.0, 10.0]

    #model = ['afhq-cat', 'afhq-dog', 'afhq-wild', 'afhq', 'ffhq', 'coco']
    N = 10
    model = ['afhq-cat']
    computation_methods = ['mean']
    reg_types = ['prior']
    interpolation_space=['noise']
    #run_model(computation_methods, model, method, clip, lam, N, reg_types, interpolation_space, max_iter, wait_time)
    
    #model = ['house', 'mountain', 'aircraft', "lion_tiger"]
    model = ['cat']
    computation_methods = ['ivp'] #['ivp', 'bvp']
    N = 100
    reg_types = ['score', 'score_naive', 'prior']
    interpolation_space=['noise']
    #run_model(computation_methods, model, method, clip, lam, N, reg_types, interpolation_space, max_iter, wait_time)
    
    reg_types = ['score']#, 'score_naive']
    interpolation_space=['noise']
    computation_methods = ['bvp'] #['ivp', 'bvp']
    N = 10
    lam =[1.0]
    run_model(computation_methods, model, method, clip, lam, N, reg_types, interpolation_space, max_iter, wait_time)
    
    return
                            
def run_model(computation_methods, model, method, clip, lam, N, reg_types, interpolation_space, max_iter, wait_time):
    
    for com_meth in computation_methods:
        for mod in model:
            for meth in method:
                time.sleep(wait_time+np.abs(np.random.normal(0.0,1.,1)[0]))
                if "ProbGEORCE" in meth or "test" in meth:
                    for reg_type in reg_types:
                        for interpolation_sp in interpolation_space:
                            for cl in clip:
                                for l in lam:
                                    generate_job(model=mod, computation_method=com_meth, method=meth, lam=l, clip=cl, N=N, max_iter=max_iter, reg_type=reg_type, interpolation_space=interpolation_sp)
                                    try:
                                        submit_job()
                                    except:
                                        time.sleep(100.0+np.abs(np.random.normal(0.0,1.,1)))
                                        try:
                                            submit_job()
                                        except:
                                            print(f"Job script with {mod}, {meth} failed!")
                else:
                    generate_job(model=mod, computation_method=com_meth, method=meth, lam=1.0, clip=0, N=N, max_iter=max_iter, reg_type="score", interpolation_space="noise")
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
