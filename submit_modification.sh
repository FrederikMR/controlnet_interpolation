    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J eagle_NoiseDiffusion_10.0
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
    
    python3 run_interpolation.py \
        --model eagle \
        --method NoiseDiffusion \
        --lam 1.0 \
        --clip 1 \
        --mu 10.0 \
        --nu 10.0 \
        --N 10 \
        --max_iter 100 \
        --ckpt_path /work3/fmry/models/controlnet/control_v11p_sd21_openpose.ckpt \
    