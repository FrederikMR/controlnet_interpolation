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
    def encode_one_step(
        self,
        x_t,
        step_idx,
        c,
        use_original_steps=False,
        unconditional_guidance_scale=1.0,
        unconditional_conditioning=None,
    ):
        """
        Perform ONE DDIM encoding step: x_t -> x_{t+1}
        """
    
        # --- timestep selection ---
        timesteps = (
            np.arange(self.ddpm_num_timesteps)
            if use_original_steps
            else self.ddim_timesteps
        )
    
        t = torch.full(
            (x_t.shape[0],),
            timesteps[step_idx],
            device=x_t.device,
            dtype=torch.long
        )
    
        # --- alpha selection ---
        if use_original_steps:
            alpha_next = self.alphas_cumprod[step_idx]
            alpha = self.alphas_cumprod_prev[step_idx]
        else:
            alpha_next = self.ddim_alphas[step_idx]
            alpha = torch.tensor(self.ddim_alphas_prev[step_idx])
    
        # --- noise prediction ---
        if unconditional_guidance_scale == 1.0:
            eps = self.model.apply_model(x_t, t, c)
        else:
            assert unconditional_conditioning is not None
    
            merged_dict = {}
            for key in unconditional_conditioning.keys():
                if unconditional_conditioning[key] is not None and c[key] is not None:
                    merged_dict[key] = [
                        torch.cat((
                            unconditional_conditioning[key][0],
                            c[key][0]
                        ))
                    ]
                else:
                    merged_dict[key] = None
    
            eps_uncond, eps_cond = torch.chunk(
                self.model.apply_model(
                    torch.cat((x_t, x_t)),
                    torch.cat((t, t)),
                    merged_dict
                ),
                2
            )
            eps = eps_uncond + unconditional_guidance_scale * (eps_cond - eps_uncond)
    
        # --- DDIM forward update ---
        x_next = (
            (alpha_next / alpha).sqrt() * x_t
            + alpha_next.sqrt()
            * (
                (1 / alpha_next - 1).sqrt()
                - (1 / alpha - 1).sqrt()
            )
            * eps
        )
    
        return x_next


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
            x_dec, _ = self.p_sample_ddim(x_dec, cond, ts, index=index, use_original_steps=use_original_steps,
                                          unconditional_guidance_scale=unconditional_guidance_scale,
                                          unconditional_conditioning=unconditional_conditioning)
            if callback: callback(i)
        return x_dec
    
    @torch.no_grad()
    def sds_score_fun(
        self,
        x: torch.Tensor,
        c,                              # positive conditioning (dict)
        t: int,                         # DDPM timestep (int)
        unconditional_conditioning,     # uncond conditioning (dict)
        negative_conditioning=None,     # optional negative conditioning (dict)
        guidance_scale=7.5,
        neg_guidance_scale=0.0,
        weight_type="one_minus_alpha",             # "none" | "alpha" | "one_minus_alpha"
    ):
        """
        Compute an SDS-style pseudo-score for Stable Diffusion / ControlNet.
    
        Returns:
            (N, C*H*W) tensor of pseudo-score vectors
        """
    
        # reshape flattened latents
        if x.ndim == 2:
            x = x.reshape(-1, 4, 96, 96)
    
        device = x.device
        N = x.shape[0]
    
        # timestep tensor
        t_batch = torch.full(
            (N,),
            t,
            device=device,
            dtype=torch.long
        )
    
        # --- ε_uncond ---
        with torch.autocast("cuda", dtype=torch.float16):
            eps_uncond = self.model.apply_model(
                x,
                t_batch,
                unconditional_conditioning
            )
    
        # --- ε_cond ---
        with torch.autocast("cuda", dtype=torch.float16):
            eps_cond = self.model.apply_model(
                x,
                t_batch,
                c
            )
    
        # CFG-style SDS gradient
        grad = eps_uncond - eps_cond
        grad = guidance_scale * grad
    
        # --- optional negative prompt ---
        if negative_conditioning is not None and neg_guidance_scale > 0:
            eps_neg = self.model.apply_model(
                x,
                t_batch,
                negative_conditioning
            )
            grad = grad + neg_guidance_scale * (eps_neg - eps_uncond)
    
        # --- optional timestep weighting ---
        if weight_type != "none":
            alpha_bar_t = self.model.alphas_cumprod[t].to(device)
            if weight_type == "alpha":
                w = alpha_bar_t
            elif weight_type == "one_minus_alpha":
                w = 1.0 - alpha_bar_t
            else:
                raise ValueError(f"Unknown weight_type: {weight_type}")
    
            grad = w * grad
    
        return grad.reshape(N, -1)
    
    @torch.no_grad()
    def score_fun(
        self,
        x: torch.Tensor,
        c,
        t: int,
        score_corrector=None,
        corrector_kwargs=None,
        unconditional_guidance_scale=1.,
        unconditional_conditioning=None,
    ):
        """
        Memory-safe SDS / CFG-style pseudo-score.
        Same signature as original score_fun.
        """
    
        batch_size = 1  # <-- tune this to fit your GPU
    
        # reshape flattened latents
        if x.ndim == 2:
            x = x.reshape(-1, 4, 96, 96)
    
        device = x.device
        N = x.shape[0]
    
        # timestep weight (computed once)
        alpha_bar_t = self.model.alphas_cumprod[t].to(device)
        w = 1.0 - alpha_bar_t
    
        scores = []
    
        for i in range(0, N, batch_size):
            x_chunk = x[i:i + batch_size]
    
            t_chunk = torch.full(
                (x_chunk.shape[0],),
                t,
                device=device,
                dtype=torch.long
            )
    
            # unconditional conditioning is required
            if unconditional_conditioning is None:
                raise ValueError("unconditional_conditioning must be provided for SDS-style score")
    
            # --- ε_uncond ---
            with torch.autocast("cuda", dtype=torch.float16):
                eps_uncond = self.model.apply_model(
                    x_chunk,
                    t_chunk,
                    unconditional_conditioning
                )
    
            # --- ε_cond ---
            with torch.autocast("cuda", dtype=torch.float16):
                eps_cond = self.model.apply_model(
                    x_chunk,
                    t_chunk,
                    c
                )
    
            # CFG-style SDS gradient
            grad = eps_uncond - eps_cond
            grad = unconditional_guidance_scale * grad
    
            # optional score corrector (rare in SDS)
            if score_corrector is not None:
                assert self.model.parameterization == "eps", "Score corrector only supported for eps models"
                with torch.autocast("cuda", dtype=torch.float16):
                    grad = score_corrector.modify_score(
                        self.model,
                        grad,
                        x_chunk,
                        t_chunk,
                        c,
                        **(corrector_kwargs or {})
                    )
    
            grad = w * grad
            grad = grad.reshape(x_chunk.shape[0], -1)
    
            scores.append(grad)
    
        return torch.cat(scores, dim=0).reshape(N, -1)

    @torch.no_grad()
    def score_fun_naive(self, x: torch.Tensor, c, t: int,
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
            with torch.autocast("cuda", dtype=torch.float16):
                eps = self.pred_eps(x_chunk, c, t_chunk,
                                    score_corrector=score_corrector, 
                                    corrector_kwargs=corrector_kwargs,
                                    unconditional_guidance_scale=unconditional_guidance_scale, 
                                    unconditional_conditioning=unconditional_conditioning,
                                    )

            # score = -eps / sqrt(1 - alpha_bar_t)
            score = -eps / denom
            score = score.reshape(x_chunk.shape[0], -1)
    
            scores.append(score)
    
        return torch.cat(scores, dim=0).reshape(N, -1)
            
 