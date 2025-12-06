import sys
from PIL import Image
import os
import cm

def run_football(N:int=10, lam:float=1.0, max_iter:int=100, inter_method:str="linear", clip:bool=False,
                  mu:float=None, nu:float=None,
                  ckpt_path:str="models/control_v11p_sd21_openpose.ckpt")->None:
    osp = os.path
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    sys.path.append(osp.expandvars('$NFS/NoiseDiffusion/controlnet'))
    
    CM = cm.ContextManager(N=N, lam=lam, max_iter=max_iter, inter_method=inter_method, clip=clip,
                           mu = mu, nu = nu,
                           ckpt_path=ckpt_path)
    img1 = Image.open('sample_imgs/football1.png').resize((768, 768))
    img2 = Image.open('sample_imgs/football2.png').resize((768, 768))
    
    prompt = 'a photo of a soccer player'
    n_prompt='text, soccer sports player'
    CM.bvp(img1, img2,  prompt=prompt, n_prompt=n_prompt, ddim_steps=200,  guide_scale=10,  out_dir='sample_results/football')
    
    return
