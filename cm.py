import pdb
import shutil
from share import *

import os
import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import math
F = torch.nn.functional

import time
import yaml
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from torch_geometry.interpolation import LinearInterpolation, SphericalInterpolation, NoiseDiffusion
from torch_geometry.prob_geodesics import ProbEuclideanGEORCE
    
class ContextManager:
    def __init__(self, 
                 N:int=10,
                 lam:float=1.0,
                 max_iter:int=100,
                 inter_method:str="linear",
                 clip:bool=False,
                 ckpt_path:str = "models/control_v11p_sd21_openpose.ckpt",
                 ):
                 
        self.inter_method = inter_method
        self.lam = lam
        self.N = N
        self.SInt = SphericalInterpolation(N=N)
        self.LInt = LinearInterpolation(N=N)
        self.PGEORCE_N = ProbEuclideanGEORCE(reg_fun = lambda x: torch.sum(x**2),
                                             init_fun=None,
                                             lam = lam,
                                             N=N,
                                             tol=1e-4,
                                             max_iter=max_iter,
                                             line_search_params = {'rho': 0.5},
                                             )
        self.PGEORCE_D = ProbEuclideanGEORCE(reg_fun = lambda x: torch.sum((torch.sum(x**2,axis=-1)-x.shape[-1])**2),
                                             init_fun=None,
                                             lam = lam,
                                             N=N,
                                             tol=1e-4,
                                             max_iter=max_iter,
                                             line_search_params = {'rho': 0.5},
                                             )
        self.clip = clip
        
        self.model = create_model('models/cldm_v21.yaml').cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        self.model.load_state_dict(load_state_dict(ckpt_path, 
                                                   location='cuda'))
        
    def noise_diffusion(self,
                        l1, 
                        l2,
                        left_image,
                        right_image,
                        noise,
                        ldm,
                        t,
                        ):
        
        shape = l1.shape
        
        l1 = l1.reshape(-1)
        l2 = l2.reshape(-1)
        left_image = left_image.reshape(-1)
        right_image = right_image.reshape(-1)
        
        s = torch.linspace(0,1,self.N+1,
                           device='cuda:0',
                           )[1:-1].reshape(-1,1)
        
        coef=2.0
        gamma=0
        #alpha=math.cos(math.radians(s*90))
        #beta=math.sin(math.radians(s*90))
        alpha = torch.cos(0.5*torch.pi*s)
        beta = torch.sin(0.5*torch.pi*s)
        
        l=alpha/beta
        
        alpha=((1-gamma*gamma)*l*l/(l*l+1))**0.5
        beta=((1-gamma*gamma)/(l*l+1))**0.5

        mu=1.2*alpha/(alpha+beta)
        nu=1.2*beta/(alpha+beta)
        
        l1=torch.clip(l1,-coef,coef)  
        l2=torch.clip(l2,-coef,coef)   
        
        print(l1.shape)
        print(l2.shape)
        print(alpha.shape)
        print(beta.shape)
        print(mu.shape)
        print(nu.shape)
        print(ldm.sqrt_one_minus_alphas_cumprod[t].shape)
        print(noise.shape)
        
        noisy_latent= torch.vmap(lambda alpha, beta, mu, nu: alpha*l1+beta*l2+\
                                 (mu-alpha)*ldm.sqrt_alphas_cumprod[t] * left_image+ \
                                     (nu-beta)*ldm.sqrt_alphas_cumprod[t] * right_image+ \
                                         gamma*noise*ldm.sqrt_one_minus_alphas_cumprod[t])(alpha, beta, mu, nu)
        
        curve=torch.clip(noisy_latent,-coef,coef)   
        
        curve = torch.vstack((l1, curve, l2))
        
        return curve.reshape(-1, *shape)

    def interpolate_new(self, img1, img2,  scale_control=1.5, prompt=None, n_prompt=None, min_steps=.3, max_steps=.55, ddim_steps=250,  guide_scale=7.5,  optimize_cond=0,  cond_lr=1e-4, bias=0, ddim_eta=0, out_dir='blend'):
        torch.manual_seed(49)
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)
        clip_str = '_clip' if self.clip else ''
        if "ProbGEORCE" in self.inter_method:
            lam = str(self.lam).replace('.', 'd')
            out_dir = ''.join((out_dir, f'/{self.inter_method}_{lam}{clip_str}'))
        else:            
            out_dir = ''.join((out_dir, f'/{self.inter_method}{clip_str}'))
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        
        if isinstance(img1, Image.Image):
            img1.save(f'{out_dir}/{0:03d}.png')
            img2.save(f'{out_dir}/{2:03d}.png')
            if img1.mode == 'RGBA':#
                    img1 = img1.convert('RGB')
            if img2.mode == 'RGBA':
                img2 = img2.convert('RGB')
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
            img2 = torch.tensor(np.array(img2)).permute(2,0,1).unsqueeze(0).cuda()
        
        ldm = self.model
        ldm.control_scales = [1] * 13

        cond1 = ldm.get_learned_conditioning([prompt])
        uncond_base = ldm.get_learned_conditioning([n_prompt])
        cond = {"c_crossattn": [cond1], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
        
        self.ddim_sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=False)#构造ddim_timesteps,赋值给timesteps
        timesteps = self.ddim_sampler.ddim_timesteps

        left_image= ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.float() / 127.5 - 1.0))
        right_image= ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.float() / 127.5 - 1.0))

        
        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, bias=bias, ddim_eta=ddim_eta, scale_control=scale_control)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))
        
        cur_step=140
        t = timesteps[cur_step]
        
        l1, _ = self.ddim_sampler.encode(left_image, cond, cur_step, 
        use_original_steps=False, return_intermediates=None,
        unconditional_guidance_scale=1, unconditional_conditioning=un_cond)
        l2, _ = self.ddim_sampler.encode(right_image, cond, cur_step, 
        use_original_steps=False, return_intermediates=None,
        unconditional_guidance_scale=1, unconditional_conditioning=un_cond)
        
        noise = torch.randn_like(left_image)
        if self.inter_method=="noise":
            l1 = ldm.sqrt_alphas_cumprod[t] * left_image + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
            l2 = ldm.sqrt_alphas_cumprod[t] * right_image + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
            noisy_curve = self.SInt(l1, l2)
        elif self.inter_method == "linear":
            noisy_curve = self.LInt(l1,l2)
        elif self.inter_method == "slerp":
            noisy_curve = self.SInt(l1,l2)
        elif self.inter_method == "noisediffusion":
            noisy_curve = self.noise_diffusion(l1, l2, left_image, right_image, noise, ldm, t)
        elif self.inter_method == "ProbGEORCE_N":
            noisy_curve = self.PGEORCE_N(l1, l2)
        elif self.inter_method == "ProbGEORCE_D":
            noisy_curve = self.PGEORCE_D(l1, l2)
            
        if self.clip:
            noisy_curve = torch.clip(noisy_curve, -2.0, 2.0)
            
        for i, noisy_latent in enumerate(noisy_curve, start=0):
            samples= self.ddim_sampler.decode(noisy_latent, cond, cur_step, # cur_step-1 / new_step-1
                unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond,
                use_original_steps=False)  
            
            image = ldm.decode_first_stage(samples)

            image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
            lam = str(self.lam).replace('.','d')
            if self.clip:
                Image.fromarray(image[0]).save(f'{out_dir}/{i}.png')
            else:
                Image.fromarray(image[0]).save(f'{out_dir}/{i}.png')

        return

