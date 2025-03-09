import sys
from PIL import Image
import os
import cm

def run_lion_tiger(N:int=10, lam:float=1.0, max_iter:int=100, inter_method:str="linear", clip:bool=False,
                   ckpt_path:str="models/control_v11p_sd21_openpose.ckpt")->None:
    osp = os.path
    os.environ['CUDA_VISIBLE_DEVICES'] = '5'
    sys.path.append(osp.expandvars('$NFS/NoiseDiffusion/controlnet'))
    
    CM = cm.ContextManager(N=N, lam=lam, max_iter=max_iter, inter_method=inter_method, clip=clip,
                           ckpt_path=ckpt_path)
    img1 = Image.open('controlnet/sample_imgs/lion_tiger1.png').resize((768, 768))
    img2 = Image.open('controlnet/sample_imgs/lion_tiger2.png').resize((768, 768))
    
    prompt = "a photo of a lion's face,a photo of a tiger's face"
    n_prompt='text, signature, logo, distorted, ugly, weird eyes, lowres, messy, weird face, lopsided, disfigured, bad art, poorly drawn, low quality, drawing, blurry, faded'
    
    CM.interpolate_new(img1, img2, prompt=prompt, n_prompt=n_prompt, ddim_steps=200, guide_scale=10,  out_dir='controlnet/sample_results/liontiger')
    
    return
