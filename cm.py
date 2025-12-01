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

from torch.distributions.chi2 import Chi2

from torch_geometry.interpolation import LinearInterpolation, SphericalInterpolation, NoiseDiffusion
from torch_geometry.prob_geodesics import ProbGEORCE_Euclidean
from torch_geometry.prob_geodesics import ProbGEORCE_Euclidean_Adaptive, ProbScoreGEORCE_Euclidean

class ContextManager:
    def __init__(self, 
                 N:int=10,
                 lam:float=1.0,
                 max_iter:int=100,
                 inter_method:str="linear",
                 clip:bool=True,
                 mu:float=None,
                 nu:float=None,
                 ckpt_path:str = "models/control_v11p_sd21_openpose.ckpt",
                 ):
                 
        self.inter_method = inter_method
        self.lam = lam
        self.N = N
        self.mu = mu
        self.nu = nu
        self.clip = clip
        self.max_iter = max_iter
        
        self.SInt = SphericalInterpolation(N=N, device="cuda:0")
        self.LInt = LinearInterpolation(N=N, device="cuda:0")
        
        self.model = create_model('models/cldm_v21.yaml').cuda()
        self.ddim_sampler = DDIMSampler(self.model)
        
        self.model.load_state_dict(load_state_dict(ckpt_path, 
                                                   location='cuda'))
        
    def score_fun(self, x: torch.Tensor, c, t: int, batch_size=1):
        """
        Compute ∇_x log p(x_t) using the DDPM formula:
            score = -eps / sqrt(1 - alpha_bar_t)
    
        Args:
            x  : (N, C,H,W) or (N, C*H*W)
            c  : conditioning dict
            t  : *actual DDPM timestep*  (int)
            batch_size: minibatch size
    
        Returns:
            (N, C*H*W) tensor of score estimates
        """
        
        # reshape flattened latents
        if x.ndim == 2:
            x = x.reshape(-1, 4, 96, 96)
    
        device = x.device
        N = x.shape[0]
    
        scores = []
    
        # proper DDPM cumulative alpha
        alpha_bar_t = self.ddim_sampler.model.alphas_cumprod[t].to(device)
        alpha_bar_t = alpha_bar_t.view(1,1,1,1)
    
        denom = torch.sqrt(1 - alpha_bar_t)
    
        for i in range(0, N, batch_size):
            x_chunk = x[i:i+batch_size]
    
            # batched timesteps for the model
            t_chunk = torch.full(
                (x_chunk.shape[0],),
                t,
                device=device,
                dtype=torch.long
            )
    
            # εθ(x_t, t)
            eps = self.ddim_sampler.pred_eps(x_chunk, c, t_chunk)
    
            # score = -eps / sqrt(1 - alpha_bar_t)
            score = -eps / denom
            score = score.reshape(x_chunk.shape[0], -1)
    
            scores.append(score)
    
        return torch.cat(scores, dim=0)

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
        noise = noise.reshape(-1)
        
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

        if self.mu is None:
            mu=1.2*alpha/(alpha+beta)
            nu=1.2*beta/(alpha+beta)
        else:
            mu = self.mu*torch.ones_like(alpha, device='cuda:0',)
            nu = self.nu*torch.ones_like(alpha, device='cuda:0',)
        
        l1=torch.clip(l1,-coef,coef)  
        l2=torch.clip(l2,-coef,coef)   
        
        noise_latent = alpha*l1+beta*l2+(mu-alpha)*ldm.sqrt_alphas_cumprod[t].reshape(-1) * left_image+(nu-beta)*ldm.sqrt_alphas_cumprod[t].reshape(-1) * right_image+gamma*noise*ldm.sqrt_one_minus_alphas_cumprod[t].reshape(-1)
        
        curve=torch.clip(noise_latent,-coef,coef)   
        
        curve = torch.vstack((l1, curve, l2))
        
        return curve.reshape(-1, *shape)
    
    def pgeorce_nd(self,
                   l1, 
                   l2,
                   left_image,
                   right_image,
                   noise,
                   ldm,
                   t,
                   ):
        
        coef=2.0
        gamma=0
        
        shape = l1.shape
        
        l1 = l1.reshape(-1)
        l2 = l2.reshape(-1)
        left_image = left_image.reshape(-1)
        right_image = right_image.reshape(-1)
        noise = noise.reshape(-1)
        
        l1=torch.clip(l1,-coef,coef)
        l2=torch.clip(l2,-coef,coef)
        
        noise_curve = self.PGEORCE(l1,l2)[1:-1]
        data_curve = self.PGEORCE(left_image, right_image)[1:-1]

        #alpha=math.cos(math.radians(s*90))
        #beta=math.sin(math.radians(s*90))
        
        s = torch.linspace(0,1,self.N+1,
                           device='cuda:0',
                           )[1:-1].reshape(-1,1)
        
        alpha = torch.cos(0.5*torch.pi*s)
        beta = torch.sin(0.5*torch.pi*s)

        if self.mu is None:
            
            l=alpha/beta
            
            alpha=((1-gamma*gamma)*l*l/(l*l+1))**0.5
            beta=((1-gamma*gamma)/(l*l+1))**0.5
            
            mu=1.2*alpha/(alpha+beta)
            nu=1.2*beta/(alpha+beta)
        else:
            mu = self.mu*torch.ones_like(alpha, device='cuda:0',)
            nu = self.nu*torch.ones_like(alpha, device='cuda:0',)
        
        
        t_cumprod = ldm.sqrt_alphas_cumprod[t].reshape(-1)
        
        noise_latent = noise_curve - t_cumprod*data_curve + \
            t_cumprod*(mu*left_image + nu * right_image)+gamma*noise*t_cumprod

        curve=torch.clip(noise_latent,-coef,coef)
        curve = torch.vstack((l1, curve, l2))
        
        return curve.reshape(-1, *shape)

    def interpolate_new(self, img1, img2,  scale_control=1.5, prompt=None, n_prompt=None, min_steps=.3, max_steps=.55, ddim_steps=250,  guide_scale=7.5,  optimize_cond=0,  cond_lr=1e-4, bias=0, ddim_eta=0, out_dir='blend'):
        torch.manual_seed(49)
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)
        base_dir = out_dir
        clip_str = '_clip' if self.clip else ''
        if "ProbGEORCE" in self.inter_method:
            lam = str(self.lam).replace('.', 'd')
            out_dir = ''.join((out_dir, f'../figures/{self.inter_method}{clip_str}_{lam}'))
        else:            
            out_dir = ''.join((out_dir, f'../figures/{self.inter_method}{clip_str}'))
            
        if self.mu is not None:
            mu_str = str(self.mu).replace('.', 'd')
            out_dir = ''.join((out_dir, f'_{mu_str}'))
            
        if self.nu is not None:
            nu_str = str(self.nu).replace('.', 'd')
            out_dir = ''.join((out_dir, f'_{nu_str}'))
            
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        
        if isinstance(img1, Image.Image):
            img1.save(f'{base_dir}/{0:03d}.png')
            img2.save(f'{base_dir}/{2:03d}.png')
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
        
        S = Chi2(len(l1.reshape(-1)))
        self.PGEORCE = ProbGEORCE_Euclidean(reg_fun = lambda x: -torch.sum(S.log_prob(torch.sum(x**2, axis=-1))),
                                           init_fun=None,
                                           lam = self.lam,
                                           N=self.N,
                                           tol=1e-4,
                                           max_iter=self.max_iter,
                                           line_search_params = {'rho': 0.5},
                                           device="cuda:0",
                                           )

        lr_rate=0.001
        beta1=0.5
        beta2=0.5
        eps=1e-8
        tol = 1e-4
        
        self.PGEORCE_Score_Noise = ProbScoreGEORCE_Euclidean(score_fun = lambda x: -self.score_fun(x,cond,cur_step),
                                                             init_fun=None,
                                                             lam=self.lam,
                                                             N=self.N,
                                                             tol=tol,
                                                             max_iter=self.max_iter,
                                                             lr_rate=lr_rate,
                                                             beta1=beta1,
                                                             beta2=beta2,
                                                             eps=eps,
                                                             device="cuda:0",
                                                             )
        
        self.PGEORCE_Score_Data = ProbScoreGEORCE_Euclidean(score_fun = lambda x: -self.score_fun(x,cond, 0),
                                                            init_fun= None,
                                                            lam=self.lam,
                                                            N=self.N,
                                                            tol=tol,
                                                            max_iter=self.max_iter,
                                                            lr_rate=lr_rate,
                                                            beta1=beta1,
                                                            beta2=beta2,
                                                            eps=eps,
                                                            device="cuda:0",
                                                            )
        
        noise = torch.randn_like(left_image)
        if self.inter_method=="Noise":
            l1 = ldm.sqrt_alphas_cumprod[t] * left_image + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
            l2 = ldm.sqrt_alphas_cumprod[t] * right_image + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
            noisy_curve = self.SInt(l1, l2)
        elif self.inter_method == "Linear":
            noisy_curve = self.LInt(l1,l2)
        elif self.inter_method == "Spherical":
            noisy_curve = self.SInt(l1,l2)
        elif self.inter_method == "NoiseDiffusion":
            noisy_curve = self.noise_diffusion(l1, l2, left_image, right_image, noise, ldm, t)
        elif self.inter_method == "ProbGEORCE":
            noisy_curve = self.PGEORCE(l1, l2)
        elif self.inter_method == "ProbGEORCE_ND":
            noisy_curve = self.pgeorce_nd(l1, l2, left_image, right_image, noise, ldm, t)
        elif self.inter_method == "ProbGEORCE_Score_Data":
            data_curve = self.PGEORCE_Score_Data(l1, l2)
            noisy_curve = None
        elif self.inter_method == "ProbGEORCE_Score_Noise":
            noisy_curve = self.PGEORCE_Score_Noise(l1, l2)
            
        if noisy_curve is not None:
            if self.clip:
                noisy_curve = torch.clip(noisy_curve, min=-2.0, max=2.0)
        
            for i, noisy_latent in enumerate(noisy_curve, start=0):
                samples= self.ddim_sampler.decode(noisy_latent, cond, cur_step, # cur_step-1 / new_step-1
                    unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond,
                    use_original_steps=False)  
                
                image = ldm.decode_first_stage(samples)
    
                image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                lam = str(self.lam).replace('.','d')
                Image.fromarray(image[0]).save(f'{out_dir}/{i}.png')
        else:
            for samples in data_curve:
                image = ldm.decode_first_stage(samples)
    
                image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                lam = str(self.lam).replace('.','d')
                Image.fromarray(image[0]).save(f'{out_dir}/{i}.png')

        return

