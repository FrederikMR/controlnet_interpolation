    #! /bin/bash
    #BSUB -q gpuv100
    #BSUB -J house_ProbGEORCE_Data
    #BSUB -n 4
    #BSUB -gpu "num=1:mode=exclusive_process"
    #BSUB -W 24:00
    #BSUB -R "span[hosts=1]"
    #BSUB -R "rusage[mem=16GB]"
    #BSUB -u fmry@dtu.dk
    #BSUB -B
    #BSUB -N
    #BSUB -o ../error_folder/error_%J.out 
    #BSUB -e ../output_folder/output_%J.err 
    
    module swap python3/3.10.12
    module swap cuda/12.0
    module swap cudnn/v8.9.1.23-prod-cuda-12.X
    
    python3 run_interpolation.py \
        --img_types house \
        --computation_method ivp \
        --n_images 2 \
        --image_size 768 \
        --target_prompt 1 \
        --method ProbGEORCE_Data \
        --lam 10.0 \
        --clip 0 \
        --mu -1.0 \
        --nu -1.0 \
        --N 100 \
        --max_iter 100 \
        --ckpt_path /work3/fmry/models/controlnet/control_v11p_sd21_openpose.ckpt \
    