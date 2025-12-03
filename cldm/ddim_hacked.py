"""SAMPLING ONLY."""

import torch
import numpy as np
from tqdm import tqdm

from ldm.modules.diffusionmodules.util import make_ddim_sampling_parameters, make_ddim_timesteps, noise_like, extract_into_tensor

    

class DDIMSampler(object):
    def __init__(self, model, schedule="linear", **kwargs):
        super().__init__()
        self.model = model
        self.ddpm_num_timesteps = model.num_timesteps
        self.schedule = schedule

    def register_buffer(self, name, attr):
        if type(attr) == torch.Tensor:
            if attr.device != torch.device("cuda"):
                attr = attr.to(torch.device("cuda"))
        setattr(self, name, attr)


    def make_schedule(self, ddim_num_steps, ddim_discretize="uniform", ddim_eta=0., verbose=True):
        self.ddim_timesteps = make_ddim_timesteps(ddim_discr_method=ddim_discretize, num_ddim_timesteps=ddim_num_steps,
                                                  num_ddpm_timesteps=self.ddpm_num_timesteps,verbose=verbose)
 
        alphas_cumprod = self.model.alphas_cumprod
        assert alphas_cumprod.shape[0] == self.ddpm_num_timesteps, 'alphas have to be defined for each timestep'
        to_torch = lambda x: x.clone().detach().to(torch.float32).to(self.model.device)

        self.register_buffer('betas', to_torch(self.model.betas))
        self.register_buffer('alphas_cumprod', to_torch(alphas_cumprod))
        self.register_buffer('alphas_cumprod_prev', to_torch(self.model.alphas_cumprod_prev))

        # calculations for diffusion q(x_t | x_{t-1}) and others
        self.register_buffer('sqrt_alphas_cumprod', to_torch(np.sqrt(alphas_cumprod.cpu())))
        self.register_buffer('sqrt_one_minus_alphas_cumprod', to_torch(np.sqrt(1. - alphas_cumprod.cpu())))
        self.register_buffer('log_one_minus_alphas_cumprod', to_torch(np.log(1. - alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recip_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu())))
        self.register_buffer('sqrt_recipm1_alphas_cumprod', to_torch(np.sqrt(1. / alphas_cumprod.cpu() - 1)))

        # ddim sampling parameters
        ddim_sigmas, ddim_alphas, ddim_alphas_prev = make_ddim_sampling_parameters(alphacums=alphas_cumprod.cpu(),
                                                                                   ddim_timesteps=self.ddim_timesteps,
                                                                                   eta=ddim_eta,verbose=verbose)
        self.register_buffer('ddim_sigmas', ddim_sigmas)
        self.register_buffer('ddim_alphas', ddim_alphas)
        self.register_buffer('ddim_alphas_prev', ddim_alphas_prev)
        self.register_buffer('ddim_sqrt_one_minus_alphas', np.sqrt(1. - ddim_alphas))
        sigmas_for_original_sampling_steps = ddim_eta * torch.sqrt(
            (1 - self.alphas_cumprod_prev) / (1 - self.alphas_cumprod) * (
                        1 - self.alphas_cumprod / self.alphas_cumprod_prev))
        self.register_buffer('ddim_sigmas_for_original_num_steps', sigmas_for_original_sampling_steps)

    @torch.no_grad()
    def sample(self,
               S,
               batch_size,
               shape,
               conditioning=None,
               callback=None,
               normals_sequence=None,
               img_callback=None,
               quantize_x0=False,
               eta=0.,
               mask=None,
               x0=None,
               temperature=1.,
               noise_dropout=0.,
               score_corrector=None,
               corrector_kwargs=None,
               verbose=True,
               x_T=None,
               log_every_t=100,
               unconditional_guidance_scale=1.,
               unconditional_conditioning=None, # this has to come in the same format as the conditioning, # e.g. as encoded tokens, ...
               dynamic_threshold=None,
               ucg_schedule=None,
               **kwargs
               ):

        if conditioning is not None:
            if isinstance(conditioning, dict):
                ctmp = conditioning[list(conditioning.keys())[0]]
                while isinstance(ctmp, list): ctmp = ctmp[0]
                cbs = ctmp.shape[0]
                if cbs != batch_size:
                    print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            elif isinstance(conditioning, list):
                for ctmp in conditioning:
                    if ctmp.shape[0] != batch_size:
                        print(f"Warning: Got {cbs} conditionings but batch-size is {batch_size}")

            else:
                if conditioning.shape[0] != batch_size:
                    print(f"Warning: Got {conditioning.shape[0]} conditionings but batch-size is {batch_size}")

        self.make_schedule(ddim_num_steps=S, ddim_eta=eta, verbose=verbose)
        # sampling
        C, H, W = shape
        size = (batch_size, C, H, W)
        print(f'Data shape for DDIM sampling is {size}, eta {eta}')

        samples, intermediates = self.ddim_sampling(conditioning, size,
                                                    callback=callback,
                                                    img_callback=img_callback,
                                                    quantize_denoised=quantize_x0,
                                                    mask=mask, x0=x0,
                                                    ddim_use_original_steps=False,
                                                    noise_dropout=noise_dropout,
                                                    temperature=temperature,
                                                    score_corrector=score_corrector,
                                                    corrector_kwargs=corrector_kwargs,
                                                    x_T=x_T,
                                                    log_every_t=log_every_t,
                                                    unconditional_guidance_scale=unconditional_guidance_scale,
                                                    unconditional_conditioning=unconditional_conditioning,
                                                    dynamic_threshold=dynamic_threshold,
                                                    ucg_schedule=ucg_schedule,
                                                    **kwargs
                                                    )
        return samples, intermediates

    @torch.no_grad()
    def ddim_sampling(self, cond, shape,
                      x_T=None, ddim_use_original_steps=False,
                      callback=None, timesteps=None, quantize_denoised=False,
                      mask=None, x0=None, img_callback=None, log_every_t=100,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None, dynamic_threshold=None,
                      ucg_schedule=None):
        # if mask is not None, overlay x0 on result
        # x0 is not used otherwise
        device = self.model.betas.device
        b = shape[0]

        if x_T is None:
            img = torch.randn(shape, device=device)
        else:
            img = x_T

        if timesteps is None:
            timesteps = self.ddpm_num_timesteps if ddim_use_original_steps else self.ddim_timesteps
        elif timesteps is not None and not ddim_use_original_steps:
            subset_end = int(min(timesteps / self.ddim_timesteps.shape[0], 1) * self.ddim_timesteps.shape[0]) - 1
            timesteps = self.ddim_timesteps[:subset_end]

        intermediates = {'x_inter': [img], 'pred_x0': [img]}
        time_range = reversed(range(0,timesteps)) if ddim_use_original_steps else np.flip(timesteps)

        
        total_steps = timesteps if ddim_use_original_steps else timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='DDIM Sampler', total=total_steps)

        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            ts = torch.full((b,), step, device=device, dtype=torch.long)

            if mask is not None:
                assert x0 is not None
                img_orig = self.model.q_sample(x0, ts)  # TODO: deterministic forward pass?
                img = img_orig * mask + (1. - mask) * img

            if ucg_schedule is not None:
                assert len(ucg_schedule) == len(time_range)
                unconditional_guidance_scale = ucg_schedule[i]

            outs = self.p_sample_ddim(img, cond, ts, index=index, use_original_steps=ddim_use_original_steps,
                                      quantize_denoised=quantize_denoised, temperature=temperature,
                                      noise_dropout=noise_dropout, score_corrector=score_corrector,
                                      corrector_kwargs=corrector_kwargs,
                                      unconditional_guidance_scale=unconditional_guidance_scale,
                                      unconditional_conditioning=unconditional_conditioning,
                                      dynamic_threshold=dynamic_threshold)
            img, pred_x0 = outs
            if callback: callback(i)
            if img_callback: img_callback(pred_x0, i)

            if index % log_every_t == 0 or index == total_steps - 1:
                intermediates['x_inter'].append(img)
                intermediates['pred_x0'].append(pred_x0)

        return img, intermediates

    @torch.no_grad()
    def p_sample_ddim(self, x, c, t, index, repeat_noise=False, use_original_steps=False, quantize_denoised=False,
                      temperature=1., noise_dropout=0., score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None,
                      dynamic_threshold=None):
        b, *_, device = *x.shape, x.device

        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            model_t = self.model.apply_model(x, t, c)
            model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)

        alphas = self.model.alphas_cumprod if use_original_steps else self.ddim_alphas
        alphas_prev = self.model.alphas_cumprod_prev if use_original_steps else self.ddim_alphas_prev
        sqrt_one_minus_alphas = self.model.sqrt_one_minus_alphas_cumprod if use_original_steps else self.ddim_sqrt_one_minus_alphas
        sigmas = self.model.ddim_sigmas_for_original_num_steps if use_original_steps else self.ddim_sigmas
        # select parameters corresponding to the currently considered timestep
        a_t = torch.full((b, 1, 1, 1), alphas[index], device=device)
        a_prev = torch.full((b, 1, 1, 1), alphas_prev[index], device=device)
        sigma_t = torch.full((b, 1, 1, 1), sigmas[index], device=device)
        sqrt_one_minus_at = torch.full((b, 1, 1, 1), sqrt_one_minus_alphas[index],device=device)

        # current prediction for x_0
        if self.model.parameterization != "v":
            pred_x0 = (x - sqrt_one_minus_at * e_t) / a_t.sqrt()
        else:
            pred_x0 = self.model.predict_start_from_z_and_v(x, t, model_output)

        if quantize_denoised:
            pred_x0, _, *_ = self.model.first_stage_model.quantize(pred_x0)

        if dynamic_threshold is not None:
            raise NotImplementedError()

        # direction pointing to x_t
        dir_xt = (1. - a_prev - sigma_t**2).sqrt() * e_t
        noise = sigma_t * noise_like(x.shape, device, repeat_noise) * temperature
        if noise_dropout > 0.:
            noise = torch.nn.functional.dropout(noise, p=noise_dropout)
        x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise
        return x_prev, pred_x0 

    def pred_eps(self, x, c, t, score_corrector=None, corrector_kwargs=None,
                      unconditional_guidance_scale=1., unconditional_conditioning=None):
        if unconditional_conditioning is None or unconditional_guidance_scale == 1.:
            model_output = self.model.apply_model(x, t, c)
        else:
            model_t = self.model.apply_model(x, t, c)
            model_uncond = self.model.apply_model(x, t, unconditional_conditioning)
            model_output = model_uncond + unconditional_guidance_scale * (model_t - model_uncond)

        if self.model.parameterization == "v":
            e_t = self.model.predict_eps_from_z_and_v(x, t, model_output)
        else:
            e_t = model_output

        if score_corrector is not None:
            assert self.model.parameterization == "eps", 'not implemented'
            e_t = score_corrector.modify_score(self.model, e_t, x, t, c, **corrector_kwargs)
        return e_t


    @torch.no_grad()
    def encode(self, x0, c, t_enc, use_original_steps=False, return_intermediates=None,
               unconditional_guidance_scale=1.0, unconditional_conditioning=None, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        num_reference_steps = timesteps.shape[0]

        assert t_enc <= num_reference_steps 
        num_steps = t_enc

        if use_original_steps:
            alphas_next = self.alphas_cumprod[:num_steps] 
            alphas = self.alphas_cumprod_prev[:num_steps]
        else:
            alphas_next = self.ddim_alphas[:num_steps]
            alphas = torch.tensor(self.ddim_alphas_prev[:num_steps])

        x_next = x0
        intermediates = []
        inter_steps = []
        for i in tqdm(range(num_steps), desc='Encoding Image'):
            t = torch.full((x0.shape[0],), timesteps[i], device=self.model.device, dtype=torch.long)
            if unconditional_guidance_scale == 1.:
                noise_pred = self.model.apply_model(x_next, t, c)
            else:
                assert unconditional_conditioning is not None
                """e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                           torch.cat((unconditional_conditioning, c))), 2)"""
                merged_dict={}
                for key in unconditional_conditioning.keys():    
                    if unconditional_conditioning[key] is not None and c[key] is not None:
                        merged_list = [torch.cat((unconditional_conditioning[key][0],c[key][0]))]  
                        merged_dict[key] = merged_list       
                    else:
                       merged_dict[key]=None
                           
                e_t_uncond, noise_pred = torch.chunk(
                    self.model.apply_model(torch.cat((x_next, x_next)), torch.cat((t, t)),
                                        merged_dict ), 2)
                noise_pred = e_t_uncond + unconditional_guidance_scale * (noise_pred - e_t_uncond)

            xt_weighted = (alphas_next[i] / alphas[i]).sqrt() * x_next
            weighted_noise_pred = alphas_next[i].sqrt() * (
                    (1 / alphas_next[i] - 1).sqrt() - (1 / alphas[i] - 1).sqrt()) * noise_pred
            x_next = xt_weighted + weighted_noise_pred
            if return_intermediates and i % (
                    num_steps // return_intermediates) == 0 and i < num_steps - 1:
                intermediates.append(x_next)
                inter_steps.append(i)
            elif return_intermediates and i >= num_steps - 2:
                intermediates.append(x_next)
                inter_steps.append(i)
            if callback: callback(i)

        out = {'x_encoded': x_next, 'intermediate_steps': inter_steps}
        if return_intermediates:
            out.update({'intermediates': intermediates})
        return x_next, out

    @torch.no_grad()
    def stochastic_encode(self, x0, t, use_original_steps=False, noise=None):
        # fast, but does not allow for exact reconstruction
        # t serves as an index to gather the correct alphas
        if use_original_steps:
            sqrt_alphas_cumprod = self.sqrt_alphas_cumprod
            sqrt_one_minus_alphas_cumprod = self.sqrt_one_minus_alphas_cumprod
        else:
            sqrt_alphas_cumprod = torch.sqrt(self.ddim_alphas)
            sqrt_one_minus_alphas_cumprod = self.ddim_sqrt_one_minus_alphas

        if noise is None:
            noise = torch.randn_like(x0)
        return (extract_into_tensor(sqrt_alphas_cumprod, t, x0.shape) * x0 +
                extract_into_tensor(sqrt_one_minus_alphas_cumprod, t, x0.shape) * noise)

    @torch.no_grad()
    def decode(self, x_latent, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
               use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = x_latent
        for i, step in enumerate(iterator):
            
            index = total_steps - i - 1
            ts = torch.full((x_latent.shape[0],), step, device=x_latent.device, dtype=torch.long)
            print("Here are the shapes:")
            print(x_dec.shape)
            print(ts.shape)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
    
    def metric(self,x_dec):
        
        def f(z):
            return self.model.decode_first_stage(z)
        
        z = x_dec.clone().requires_grad_(True)
        
        # Flatten z so the identity basis vectors make sense
        z_flat = z.view(-1)
        N = z_flat.numel()
        
        JTJ = torch.zeros(N, N, device=z.device)
        
        for i in range(N):
            # basis vector
            v = torch.zeros_like(z_flat)
            v[i] = 1.0
            v = v.view_as(z)
        
            # J v
            Jv = torch.autograd.functional.jvp(f, (z,), (v,))[1]
        
            # Jᵀ(J v)
            grad = torch.autograd.grad(Jv, z, grad_outputs=Jv, retain_graph=True)[0]
            JTJ[:, i] = grad.view(-1)
            
        return JTJ
    
    def ip_grad(self,x_dec):
        
        def f(z):
            return self.model.decode_first_stage(z)
        
        z = x_dec.clone().requires_grad_(True)
        v = torch.randn_like(z)   # your chosen vector
        
        # Step 1: J v
        Jv = torch.autograd.functional.jvp(f, (z,), (v,))[1]
        
        # Step 2: Jᵀ (J v)
        u = torch.autograd.grad(Jv, z, grad_outputs=Jv, create_graph=True)[0]
        
        # Step 3: vᵀ u
        vJTJv = (u * v).sum()
        
        vJTJv.backward()   # gradient stored in z.grad
        grad_vJTJv = z.grad
        
        return grad_vJTJv

    def test(self, curve, cond, t_start, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
                   use_original_steps=False, callback=None):

        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        x_dec = curve
        for i, step in enumerate(iterator):
            
            index = total_steps - i - 1
            ts = torch.full((curve.shape[0],), step, device=curve.device, dtype=torch.long)
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return self.model.decode_first_stage(x_dec)
    
    @torch.no_grad()
    def score_fun(self, x: torch.Tensor, c, t: int,
                  score_corrector=None, corrector_kwargs=None,
                  unconditional_guidance_scale=1., unconditional_conditioning=None):
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
        batch_size = 1
        
        # reshape flattened latents
        if x.ndim == 2:
            x = x.reshape(-1, 4, 96, 96)
    
        device = x.device
        N = x.shape[0]
    
        scores = []
    
        # proper DDPM cumulative alpha
        alpha_bar_t = self.model.alphas_cumprod[t].to(device)
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
            eps = self.pred_eps(x_chunk, c, t_chunk)
    
            # score = -eps / sqrt(1 - alpha_bar_t)
            score = -eps / denom
            score = score.reshape(x_chunk.shape[0], -1)
    
            scores.append(score)
    
        return torch.cat(scores, dim=0).reshape(N, -1)
    
    @torch.no_grad()
    def iterative_geodesics(self, curve, cond, t_start, lam=1.0, unconditional_guidance_scale=1.0, unconditional_conditioning=None,
                            use_original_steps=False, callback=None):
        
        lr_rate=0.01
        beta1=0.5
        beta2=0.5
        eps=1e-8
        tol = 1e-4
        from torch_geometry.prob_geodesics import ProbScoreGEORCE_Euclidean
        
        timesteps = np.arange(self.ddpm_num_timesteps) if use_original_steps else self.ddim_timesteps
        timesteps = timesteps[:t_start]

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        print(f"Running DDIM Sampling with {total_steps} timesteps")

        iterator = tqdm(time_range, desc='Decoding image', total=total_steps)
        curve = curve.squeeze()
    
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            curve = curve.reshape(-1,4,96,96)
    
            # --- DDIM step (reduced memory)
            with torch.inference_mode(), torch.cuda.amp.autocast(dtype=torch.float16):
                new_curve = []
                for val in curve:
                    val = val.reshape(1,4,96,96)
                    ts = torch.full((val.shape[0],), step, device=val.device, dtype=torch.long)
                    print("Here are the shapes:")
                    print(val.shape)
                    print(ts.shape)
                    update, _ = self.p_sample_ddim(val, cond, ts, index=index, use_original_steps=use_original_steps,
                                                   unconditional_guidance_scale=unconditional_guidance_scale,
                                                   unconditional_conditioning=unconditional_conditioning)
                    new_curve.append(val)
            curve = torch.concatenate(new_curve, axis=0).squeeze()
    
            # Prepare interior without capturing curve
            interior = curve[1:-1].detach().reshape(len(curve[1:-1]), -1)
    
            # Create geodesic object
            PGEORCE = ProbScoreGEORCE_Euclidean(
                score_fun=lambda x: -self.score_fun(
                    x, cond, step,
                    unconditional_guidance_scale=unconditional_guidance_scale,
                    unconditional_conditioning=unconditional_conditioning,
                    use_original_steps=use_original_steps
                ),
                init_fun = lambda x,y,t, pts=interior: pts,
                lam=lam,
                N=10,
                tol=tol,
                max_iter=5,
                lr_rate=lr_rate,
                beta1=beta1,
                beta2=beta2,
                eps=eps,
                device="cuda:0",
            )
    
            # Geodesic (also reduced memory)
            with torch.inference_mode():
                curve = PGEORCE(curve[0], curve[-1])
    
            # Clean up
            del PGEORCE, interior
            torch.cuda.empty_cache()
    
            if callback:
                callback(i)
    
        return curve
            
            
import torch
from torch.autograd.functional import jvp


class JacobianOps:
    """
    Efficient JVP, VJP, JTJv, v^T JTJ v, and full JTJ construction
    for a decoder f(z) such as model.decode_first_stage.
    """

    def __init__(self, decoder):
        """
        decoder: a function f(z) returning the decoded image.
        """
        self.f = decoder

    # ---------- Basic Operators ----------

    @torch.no_grad()
    def Jv(self, z, v):
        """Compute J(z) * v (JVP) without building graph."""
        return jvp(self.f, (z,), (v,))[1]

    def Jv_graph(self, z, v):
        """JVP but keeps autograd graph for higher-order derivatives."""
        return jvp(self.f, (z,), (v,), create_graph=True)[1]

    def JTv(self, z, v_out):
        """Compute J(z)^T * v_out (VJP)."""
        return torch.autograd.grad(
            self.f(z),
            z,
            grad_outputs=v_out,
            create_graph=True
        )[0]

    # ---------- Main Operators ----------

    def JTJv(self, z, v):
        """
        Compute (J^T J) v using JVP + VJP.
        Keeps graph for differentiability.
        """
        Jv = self.Jv_graph(z, v)    # J v
        JTJv = torch.autograd.grad(
            Jv, z,
            grad_outputs=Jv,
            create_graph=True
        )[0]
        return JTJv

    def vJTJv(self, z, v):
        """
        Compute scalar v^T J^T J v.
        Fully differentiable w.r.t. z.
        """
        JTJv_val = self.JTJv(z, v)
        return (JTJv_val * v).sum()

    # ---------- FULL MATRIX (NO GRAPH) ----------
    
    @torch.no_grad()
    def compute_JTJ(self, z):
        """
        Explicitly compute the matrix J^T J at z.
        No gradient graph is created.
        Suitable only for small latent spaces.

        Returns: tensor of shape (N, N) where N = z.numel().
        """
        z_flat = z.reshape(-1)
        N = z_flat.numel()
        JTJ = torch.zeros(N, N, device=z.device, dtype=z.dtype)

        for i in range(N):
            # basis vector e_i
            v = torch.zeros_like(z_flat)
            v[i] = 1.0
            v = v.view_as(z)

            # compute (J^T J) e_i
            JTJv_i = self.JTJv(z, v)  # J^T J v
            JTJ[:, i] = JTJv_i.reshape(-1)

        return 
    
    def vJTJv_batch(self, z, v):
        """
        Vectorized batch computation of sum_i v_i^T J_i^T J_i v_i.
    
        Returns one scalar suitable for backward().
        """
        # Step 1: J v for each batch element
        Jv = jvp(self.f, (z,), (v,), create_graph=True)[1]   # same shape as f(z)
    
        # Step 2: J^T (J v)
        JTJv = torch.autograd.grad(
            self.f(z),
            z,
            grad_outputs=Jv,
            create_graph=True
        )[0]
    
        # Step 3: sum_i v_i^T JTJv_i
        return (JTJv * v).sum()




