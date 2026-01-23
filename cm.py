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
from torch.func import grad

import time
import yaml
from cldm.model import create_model, load_state_dict
from cldm.ddim_hacked import DDIMSampler

from torch.distributions.chi2 import Chi2

from torch_geometry.interpolation import (
    LinearInterpolation, 
    SphericalInterpolation, 
    NoiseDiffusion,
    )

from torch_geometry.prob_geodesics import (
    ProbGEORCE_Euclidean,
    ProbScoreGEORCE_Euclidean,
    )

from torch_geometry.prob_means import (
    ProbGEORCEFM_Euclidean, 
    ProbScoreGEORCEFM_Euclidean,
    )

from torch_geometry.manifolds import (
    nEuclidean, 
    LambdaManifold,
    )

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
                 num_images:int=None,
                 seed:int=2712,
                 reg_type:str="score",
                 interpolation_space:str="noise",
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
        
        if num_images is None:
            self.step_save = 1
        else:
            self.step_save = max(int(N / num_images), 1)
        self.seed = seed
        
        self.reg_type = reg_type
        self.interpolation_space = interpolation_space
        
    def get_reg_fun(self,
                    dimension,
                    latent_shape,
                    cur_step,
                    cond,
                    un_cond,
                    guide_scale,
                    method:str="bvp",
                    ):
        
        if self.interpolation_space == "data":
            cur_step = 0
            
        if self.reg_type == "score":
            score_fun = lambda x: self.ddim_sampler.score_fun(x.reshape(-1,*latent_shape),
                                                              cond,
                                                              cur_step,
                                                              score_corrector=None,
                                                              corrector_kwargs=None,
                                                              unconditional_guidance_scale=guide_scale,
                                                              unconditional_conditioning=un_cond,
                                                              ).reshape(*x.shape)
        elif self.reg_type == "score_naive":
            score_fun = lambda x: self.ddim_sampler.score_fun_naive(x.reshape(-1,*latent_shape),
                                                                    cond,
                                                                    cur_step,
                                                                    score_corrector=None,
                                                                    corrector_kwargs=None,
                                                                    unconditional_guidance_scale=guide_scale,
                                                                    unconditional_conditioning=un_cond,
                                                                    ).reshape(*x.shape)
        elif self.reg_type == "prior":
            S = Chi2(df=float(dimension))
            
            prior_reg_fun = lambda x: -S.log_prob(torch.sum(x.reshape(-1,dimension)**2, axis=-1)).sum()
            
            score_fun = grad(prior_reg_fun)
        elif self.reg_type == "score_naive_with_prior":
            score_fun1 = lambda x: self.ddim_sampler.score_fun_naive(x.reshape(-1,*latent_shape),
                                                                    cond,
                                                                    cur_step,
                                                                    score_corrector=None,
                                                                    corrector_kwargs=None,
                                                                    unconditional_guidance_scale=guide_scale,
                                                                    unconditional_conditioning=un_cond,
                                                                    ).reshape(*x.shape)
            
            S = Chi2(df=float(dimension))
            
            reg_fun = lambda x: -S.log_prob(torch.sum(x.reshape(-1,dimension)**2, axis=-1)).sum()
            
            score_fun2 = grad(reg_fun)
            
            score_fun = lambda x: score_fun1(x) + score_fun2(x)
            
            
        else:
            raise ValueError(f"Invalid reg_type: {self.reg_type}")
            
            
        tol = 1e-4
        lr_rate = 0.001
        beta1 = 0.5
        beta2 = 0.5
        eps = 1e-8
        device = "cuda:0"
        if method == "ivp":
            M = nEuclidean(dim=dimension)
            Mlambda = LambdaManifold(M=M, gradS=lambda x: score_fun(x.reshape(-1,dimension)).squeeze(), S=None, lam=self.lam)
            return lambda x,v: Mlambda.Exp_ode_Euclidean(x.reshape(1,-1), v.reshape(1,-1), T=self.N).reshape(-1,1,*latent_shape)
        elif method == "bvp":
            if self.reg_type == "prior":
                return ProbGEORCE_Euclidean(reg_fun = prior_reg_fun,
                                            init_fun=None,
                                            lam=self.lam,
                                            N=self.N,
                                            tol=tol,
                                            max_iter=self.max_iter,
                                            line_search_params={'rho': 0.5},
                                            device=device,
                                            )
            else:
                return ProbScoreGEORCE_Euclidean(score_fun = score_fun,
                                                 init_fun=None,
                                                 lam = self.lam,
                                                 N=self.N,
                                                 tol=tol,
                                                 max_iter=self.max_iter,
                                                 lr_rate=lr_rate,
                                                 beta1=beta1,
                                                 beta2=beta2,
                                                 eps=eps,
                                                 device=device,
                                                 )
        elif method == "mean":
            if self.reg_type == "prior":
                return ProbGEORCEFM_Euclidean(reg_fun=prior_reg_fun,
                                              init_fun=None,
                                              lam=self.lam,
                                              N_grid = self.N,
                                              tol=tol,
                                              max_iter=self.max_iter,
                                              line_search_params={'rho': 0.5},
                                              device=device,
                                              )
            else:
                return ProbScoreGEORCEFM_Euclidean(score_fun = score_fun,
                                                   init_fun= None,
                                                   lam=self.lam,
                                                   N_grid=self.N,
                                                   tol=tol,
                                                   max_iter=self.max_iter,
                                                   lr_rate=lr_rate,
                                                   beta1=beta1,
                                                   beta2=beta2,
                                                   eps=eps,
                                                   device=device,
                                                   )
        else:
            raise ValueError(f"Invalid method: {method}")
        
        return 
        
    def prompt_strength(self,
                        t, 
                        noisy_curve,
                        ):
        
        if t >= len(noisy_curve) - 1:
            return 1.0
        elif t == 0:
            return 0.0
        else:
            total_error = torch.cumsum(torch.linalg.norm((noisy_curve[1:]-noisy_curve[:-1]).reshape(len(noisy_curve)-1,-1), axis=1), axis=0)
            return total_error[t-1]/total_error[-1]
        
    def create_out_dir(self,
                       out_dir:str,
                       method_name:str,
                       )->str:
        
        base_dir = out_dir
        clip_str = '_clip' if self.clip else ''
        if "ProbGEORCE" == self.inter_method:
            inter_method = f"ProbGEORCE_{self.reg_type}_{self.interpolation_space}"
            lam = str(self.lam).replace('.', 'd')
            out_dir = ''.join((out_dir, f'/{method_name}/{inter_method}{clip_str}_{lam}'))
        else:            
            out_dir = ''.join((out_dir, f'/{method_name}/{self.inter_method}{clip_str}'))
            
        if self.mu is not None:
            mu_str = str(self.mu).replace('.', 'd')
            out_dir = ''.join((out_dir, f'_{mu_str}'))
            
        if self.nu is not None:
            nu_str = str(self.nu).replace('.', 'd')
            out_dir = ''.join((out_dir, f'_{nu_str}'))
            
        shutil.rmtree(out_dir, ignore_errors=True)
        os.makedirs(out_dir)
        
        return base_dir, out_dir
    
    def images_to_tensors_raw(self,
                              imgs, 
                              base_dir:str,
                              device='cuda',
                              ):
        tensors = []
        
        for counter, img in enumerate(imgs, start=0):
            img.save(f'{base_dir}/{0:03d}_{counter}.png')
            if img.mode == 'RGBA':#
                    img = img.convert('RGB')
            img = torch.tensor(np.array(img)).permute(2,0,1).unsqueeze(0).cuda()
            tensors.append(img)
        
        return tensors
    
    def encode_decode_images(self,
                             ldm,
                             img,
                             cond,
                             uncond_base,
                             cur_step,
                             guide_scale,
                             ):
        
        l1, _ = self.ddim_sampler.encode(img, cond, cur_step, 
        use_original_steps=False, return_intermediates=None,
        unconditional_guidance_scale=guide_scale, unconditional_conditioning=uncond_base)
        
        cond = {"c_crossattn": [cond], 'c_concat': None}
        un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
    
        # ---- Your original decode ----
        img_data = self.ddim_sampler.decode(
            l1,
            cond,
            cur_step,
            unconditional_guidance_scale=guide_scale,
            unconditional_conditioning=un_cond,
            use_original_steps=False
        )
        
        return img_data
    
    def sample_images(self,
                      ldm,
                      noisy_curve,
                      cond_neutral,
                      uncond_base,
                      cur_step,
                      guide_scale,
                      out_dir,
                      cond_target=None,
                      ):
        
        if self.clip:
            noisy_curve = torch.clip(noisy_curve, min=-2.0, max=2.0)
        
        for i, noisy_latent in enumerate(noisy_curve, start=0):
            if (i % self.step_save == 0) or (i == 0) or (i==len(noisy_curve)-1):
                if cond_target is not None:
                    # ---- NEW: smooth prompt transition ----
                    alpha = self.prompt_strength(i, noisy_curve)
                
                    cond_blend = cond_neutral * (1 - alpha) + cond_target * alpha
                else:
                    cond_blend = cond_neutral
                
                cond = {"c_crossattn": [cond_blend], 'c_concat': None}
                un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
            
                # ---- Your original decode ----
                samples = self.ddim_sampler.decode(
                    noisy_latent,
                    cond,
                    cur_step,
                    unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond,
                    use_original_steps=False
                )
            
                image = ldm.decode_first_stage(samples)
                image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                Image.fromarray(image[0]).save(f'{out_dir}/{i}.png')
        
        return
    
    def sample_multi_images(self,
                            ldm,
                            noisy_curves,
                            cond_neutral,
                            uncond_base,
                            cur_step,
                            guide_scale,
                            out_dir,
                            cond_target=None,
                            ):
        
        if self.clip:
            noisy_curve = torch.clip(noisy_curves, min=-2.0, max=2.0)
        
        for j, noisy_curve in enumerate(noisy_curves, start=0):
            for i, noisy_latent in enumerate(noisy_curve, start=0):
                if (i % self.step_save == 0) or (i == 0) or (i==len(noisy_curve)-1):
                    if cond_target is not None:
                        # ---- NEW: smooth prompt transition ----
                        alpha = self.prompt_strength(i, noisy_curve)
                    
                        cond_blend = cond_neutral * (1 - alpha) + cond_target * alpha
                    else:
                        cond_blend = cond_neutral
                    
                    cond = {"c_crossattn": [cond_blend], 'c_concat': None}
                    un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
                
                    # ---- Your original decode ----
                    samples = self.ddim_sampler.decode(
                        noisy_latent,
                        cond,
                        cur_step,
                        unconditional_guidance_scale=guide_scale,
                        unconditional_conditioning=un_cond,
                        use_original_steps=False
                    )
                
                    image = ldm.decode_first_stage(samples)
                    image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                    Image.fromarray(image[0]).save(f'{out_dir}/{j}_{i}.png')
        
        return
    
    def sample_data_images(self,
                           ldm,
                           data_curves,
                           noise,
                           cond_neutral,
                           uncond_base,
                           cur_step,
                           guide_scale,
                           out_dir,
                           cond_target=None,
                           ):
        
        for i, data_latent in enumerate(data_curves, start=0):
            if (i % self.step_save == 0) or (i == 0) or (i==len(data_curves)-1):
                
                timesteps = self.ddim_sampler.ddim_timesteps
                if cur_step == len(timesteps):
                    t = timesteps[cur_step-1]
                else:
                    t = timesteps[cur_step]
                noisy_latent = ldm.sqrt_alphas_cumprod[t] * data_latent + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                
                if cond_target is not None:
                    # ---- NEW: smooth prompt transition ----
                    alpha = self.prompt_strength(i, data_curves)
                
                    cond_blend = cond_neutral * (1 - alpha) + cond_target * alpha
                else:
                    cond_blend = cond_neutral
                
                cond = {"c_crossattn": [cond_blend], 'c_concat': None}
                un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
            
                # ---- Your original decode ----
                samples = self.ddim_sampler.decode(
                    noisy_latent,
                    cond,
                    cur_step,
                    unconditional_guidance_scale=guide_scale,
                    unconditional_conditioning=un_cond,
                    use_original_steps=False
                )
            
                image = ldm.decode_first_stage(samples)
                image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                Image.fromarray(image[0]).save(f'{out_dir}/{i}.png')
        
        return
    
    def sample_data_multi_images(self,
                                 ldm,
                                 data_curves,
                                 noise,
                                 cond_neutral,
                                 uncond_base,
                                 cur_step,
                                 guide_scale,
                                 out_dir,
                                 cond_target=None,
                                 ):
        
        for j, data_curve in enumerate(data_curves, start=0):
            for i, data_latent in enumerate(data_curve, start=0):
                if (i % self.step_save == 0) or (i == 0) or (i==len(data_curve)-1):
                    
                    timesteps = self.ddim_sampler.ddim_timesteps
                    if cur_step == len(timesteps):
                        t = timesteps[cur_step-1]
                    else:
                        t = timesteps[cur_step]
                    noisy_latent = ldm.sqrt_alphas_cumprod[t] * data_latent + ldm.sqrt_one_minus_alphas_cumprod[t] * noise
                    
                    if cond_target is not None:
                        # ---- NEW: smooth prompt transition ----
                        alpha = self.prompt_strength(i, data_curve)
                    
                        cond_blend = cond_neutral * (1 - alpha) + cond_target * alpha
                    else:
                        cond_blend = cond_neutral
                    
                    cond = {"c_crossattn": [cond_blend], 'c_concat': None}
                    un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
                
                    # ---- Your original decode ----
                    samples = self.ddim_sampler.decode(
                        noisy_latent,
                        cond,
                        cur_step,
                        unconditional_guidance_scale=guide_scale,
                        unconditional_conditioning=un_cond,
                        use_original_steps=False
                    )
                
                    image = ldm.decode_first_stage(samples)
                    image = (image.permute(0, 2, 3, 1) * 127.5 + 127.5).cpu().numpy().clip(0, 255).astype(np.uint8)
                    Image.fromarray(image[0]).save(f'{out_dir}/{j}_{i}.png')
        
        return

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

    def bvp(self, 
            img1, 
            img2,  
            scale_control=1.5,
            prompt=None, 
            n_prompt=None, 
            min_steps=.3,
            max_steps=.55, 
            ddim_steps=250,  
            guide_scale=7.5,  
            optimize_cond=0,  
            cond_lr=1e-4, 
            bias=0, 
            ddim_eta=0, 
            out_dir='blend',
            ):
        torch.manual_seed(self.seed)
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)
        base_dir, out_dir = self.create_out_dir(out_dir, "bvp")
        
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

        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        with torch.no_grad():
            cond1 = ldm.get_learned_conditioning([prompt])
            uncond_base = ldm.get_learned_conditioning([n_prompt])
            cond = {"c_crossattn": [cond1], 'c_concat': None}
            un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
        
        self.ddim_sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=False)#构造ddim_timesteps,赋值给timesteps

        left_image= ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.float() / 127.5 - 1.0))
        right_image= ldm.get_first_stage_encoding(ldm.encode_first_stage(img2.float() / 127.5 - 1.0))

        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, bias=bias, ddim_eta=ddim_eta, scale_control=scale_control)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))
        
        cur_step = ddim_steps#140 
        
        l1, _ = self.ddim_sampler.encode(left_image, cond, cur_step, 
        use_original_steps=False, return_intermediates=None,
        unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
        l2, _ = self.ddim_sampler.encode(right_image, cond, cur_step, 
        use_original_steps=False, return_intermediates=None,
        unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
        latent_shape = l1.shape
        
        noise = torch.randn_like(left_image)
        if self.inter_method=="Noise":
            timesteps = self.ddim_sampler.ddim_timesteps
            t = timesteps[cur_step]
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
            dimension = len(l1.reshape(-1))
            latent_shape = l1.shape[1:]
            bvp_method = self.get_reg_fun(dimension=dimension, 
                                          latent_shape=latent_shape,
                                          cur_step=cur_step, 
                                          cond=cond, 
                                          un_cond=un_cond,
                                          guide_scale=guide_scale,
                                          method = "bvp"
                                          )
            
            if self.interpolation_space == "noise":
                noisy_curve = bvp_method(l1, l2)
            elif self.interpolation_space == "data":
                
                left_image = self.encode_decode_images(ldm, left_image, cond, uncond_base, cur_step, guide_scale)
                right_image = self.encode_decode_images(ldm, right_image, cond, uncond_base, cur_step, guide_scale)
                
                data_curve = bvp_method(left_image, right_image)
                
                self.sample_data_images(ldm, 
                                        data_curve, 
                                        torch.randn_like(left_image),
                                        cond1, 
                                        uncond_base, 
                                        cur_step, 
                                        guide_scale, 
                                        out_dir,
                                        )
                
                return
            else:
                raise ValueError(f"Invalid interpolation space: {self.interpolation_space}")
        elif self.inter_method == "ProbGEORCE_ND":
            noisy_curve = self.pgeorce_nd(l1, l2, left_image, right_image, noise, ldm, t)
            
        self.sample_images(ldm, 
                           noisy_curve, 
                           cond1, 
                           uncond_base, 
                           cur_step, 
                           guide_scale, 
                           out_dir,
                           )
    
    def ivp(self, 
            img1, 
            scale_control=1.5,
            prompt_neutral=None, 
            prompt_target=None, 
            n_prompt=None, 
            min_steps=.3,
            max_steps=.55, 
            ddim_steps=250,  
            guide_scale=7.5,  
            optimize_cond=0,  
            cond_lr=1e-4, 
            bias=0, 
            ddim_eta=0, 
            out_dir='blend',
            ):
        torch.manual_seed(self.seed)
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)
        base_dir, out_dir = self.create_out_dir(out_dir, "ivp")
        
        if isinstance(img1, Image.Image):
            img1.save(f'{base_dir}/{0:03d}.png')
            if img1.mode == 'RGBA':#
                    img1 = img1.convert('RGB')
            img1 = torch.tensor(np.array(img1)).permute(2,0,1).unsqueeze(0).cuda()
        
        ldm = self.model
        ldm.control_scales = [1] * 13
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        with torch.no_grad():
            cond1 = ldm.get_learned_conditioning([prompt_neutral])
            uncond_base = ldm.get_learned_conditioning([n_prompt])
            cond = {"c_crossattn": [cond1], 'c_concat': None}
            un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
        
        self.ddim_sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=False)#构造ddim_timesteps,赋值给timesteps

        left_image= ldm.get_first_stage_encoding(ldm.encode_first_stage(img1.float() / 127.5 - 1.0))

        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt_neutral=prompt_neutral, 
                      prompt_target=prompt_target,
                      n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, bias=bias, ddim_eta=ddim_eta, scale_control=scale_control)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))
        
        cur_step = ddim_steps#140
        
        l1, _ = self.ddim_sampler.encode(left_image, cond, cur_step, 
        use_original_steps=False, return_intermediates=None,
        unconditional_guidance_scale=guide_scale, unconditional_conditioning=un_cond)
        latent_shape = l1.shape
        
        # Precompute conditioning
        with torch.no_grad():
            cond_target  = ldm.get_learned_conditioning([prompt_target])
            cond_neutral = ldm.get_learned_conditioning([prompt_neutral])
            uncond_base  = ldm.get_learned_conditioning([n_prompt])
            
        if self.inter_method == "ProbGEORCE":
            dimension = len(l1.reshape(-1))
            latent_shape = l1.shape[1:]
            ivp_method = self.get_reg_fun(dimension=dimension, 
                                          latent_shape=latent_shape,
                                          cur_step=cur_step, 
                                          cond=cond, 
                                          un_cond=un_cond,
                                          guide_scale=guide_scale,
                                          method = "ivp"
                                          )
            
            if self.interpolation_space == "noise":
                v0 = torch.randn_like(l1)
                noisy_curve = ivp_method(l1, v0)
            elif self.interpolation_space == "data":
                
                left_image = self.encode_decode_images(ldm, left_image, cond, uncond_base, cur_step, guide_scale)

                v0 = torch.randn_like(left_image)
                data_curve = ivp_method(left_image, v0)
                
                self.sample_data_images(ldm, 
                                        data_curve, 
                                        torch.randn_like(left_image),
                                        cond_neutral, 
                                        uncond_base, 
                                        cur_step, 
                                        guide_scale, 
                                        out_dir,
                                        cond_target,
                                        )
                
                return
                
            else:
                raise ValueError(f"Invalid interpolation space: {self.interpolation_space}")
            
        self.sample_images(ldm, 
                           noisy_curve, 
                           cond_neutral, 
                           uncond_base, 
                           cur_step, 
                           guide_scale, 
                           out_dir,
                           cond_target,
                           )
            
        return
    
    def mean(self, 
             imgs, 
             scale_control=1.5,
             prompt=None, 
             n_prompt=None, 
             min_steps=.3,
             max_steps=.55, 
             ddim_steps=250,  
             guide_scale=7.5,  
             optimize_cond=0,  
             cond_lr=1e-4, 
             bias=0, 
             ddim_eta=0, 
             out_dir='blend',
             ):
        torch.manual_seed(self.seed)
        if min_steps < 1:
            min_steps = int(ddim_steps * min_steps)
        if max_steps < 1:
            max_steps = int(ddim_steps * max_steps)
        base_dir, out_dir = self.create_out_dir(out_dir, "mean")
        
        imgs = self.images_to_tensors_raw(imgs, base_dir, "cuda")
        
        ldm = self.model
        ldm.control_scales = [1] * 13

        with torch.no_grad():
            cond1 = ldm.get_learned_conditioning([prompt])
            uncond_base = ldm.get_learned_conditioning([n_prompt])
            cond = {"c_crossattn": [cond1], 'c_concat': None}
            un_cond = {"c_crossattn": [uncond_base], 'c_concat': None}
        
        self.ddim_sampler.make_schedule(ddim_steps, ddim_eta=ddim_eta, verbose=False)#构造ddim_timesteps,赋值给timesteps
        
        img_first_stage_encodings = [ldm.get_first_stage_encoding(ldm.encode_first_stage(img.float() / 127.5 - 1.0)) for img in imgs]
        latent_shape = img_first_stage_encodings[0].shape

        kwargs = dict(cond_lr=cond_lr, cond_steps=optimize_cond, prompt=prompt, n_prompt=n_prompt, ddim_steps=ddim_steps, guide_scale=guide_scale, bias=bias, ddim_eta=ddim_eta, scale_control=scale_control)
        yaml.dump(kwargs, open(f'{out_dir}/args.yaml', 'w'))
        
        cur_step = ddim_steps#140
        
        img_encoded = torch.concatenate([self.ddim_sampler.encode(img, 
                                                                  cond, 
                                                                  cur_step, 
                                                                  use_original_steps=False, 
                                                                  return_intermediates=None,
                                                                  unconditional_guidance_scale=guide_scale, 
                                                                  unconditional_conditioning=un_cond)[0] for img in img_first_stage_encodings], axis=0)
        
        if self.inter_method == "ProbGEORCE":
            dimension = len(img_encoded[0].reshape(-1))
            latent_shape = img_encoded[0].shape
            mean_method = self.get_reg_fun(dimension=dimension, 
                                          latent_shape=latent_shape,
                                          cur_step=cur_step, 
                                          cond=cond, 
                                          un_cond=un_cond,
                                          guide_scale=guide_scale,
                                          method = "mean"
                                          )
            
            if self.interpolation_space == "noise":
                noisy_mean, noisy_curve = mean_method(img_encoded)
                noisy_curve = noisy_curve.reshape(len(noisy_curve),-1,1,*latent_shape)
            elif self.interpolation_space == "data":
                print(type(img_first_stage_encodings[0]))
                img_data_space = torch.stack([self.encode_decode_images(ldm, img, cond, uncond_base, cur_step, guide_scale) for img in img_first_stage_encodings])
                #img_data_space = torch.stack([torch.tensor(img) for img in img_first_stage_encodings])
                data_mean, data_curve = mean_method(img_data_space)
                
                self.sample_data_multi_images(ldm, 
                                              data_curve, 
                                              torch.randn_like(img_first_stage_encodings[0]),
                                              cond1, 
                                              uncond_base, 
                                              cur_step, 
                                              guide_scale, 
                                              out_dir,
                                              )
                
                return
            else:
                raise ValueError(f"Invalid interpolation space: {self.interpolation_space}")

        self.sample_multi_images(ldm, 
                                 noisy_curve, 
                                 cond1, 
                                 uncond_base, 
                                 cur_step, 
                                 guide_scale, 
                                 out_dir,
                                 )

