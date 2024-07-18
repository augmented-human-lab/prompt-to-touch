import matplotlib.pyplot as plt
import numpy as np
import sys
import torch
from tqdm import tqdm

import sys
sys.path.insert(0, '../')
from audioldm2 import seed_everything, make_batch_for_text_to_audio, build_model
from audioldm2.latent_diffusion.modules.diffusionmodules.util import noise_like
from audioldm2.latent_diffusion.models.ddim import DDIMSampler
from audioldm2.utilities import *
from audioldm2.utilities.audio import *
from audioldm2.utilities.data import *

def get_model(model_name, duration=10.0, latent_t_per_second=25.6):
    print('Loading model')
    
    latent_diffusion = build_model(model_name=model_name)
    latent_diffusion.latent_t_size = int(duration * latent_t_per_second)

    print('Model loaded')
    return latent_diffusion

def sample(latent_diffusion, target_text, batch_size=1, ddim_steps=100, \
                                    guidance_scale=3.0, random_seed=42, \
                                    disable_tqdmoutput=False):

    with torch.no_grad():
        seed_everything(int(random_seed))
        x_init = torch.randn((1, 8, 256, 16), device="cuda")

        uncond_dict = {}
        for key in latent_diffusion.cond_stage_model_metadata:
            model_idx = latent_diffusion.cond_stage_model_metadata[key]["model_idx"]
            uncond_dict[key] = latent_diffusion.cond_stage_models[
                model_idx
            ].get_unconditional_condition(batch_size)

        target_cond_batch = make_batch_for_text_to_audio(target_text, transcription="", waveform=None, batchsize=batch_size)
        _, c = latent_diffusion.get_input(target_cond_batch, latent_diffusion.first_stage_key,unconditional_prob_cfg=0.0)  # Do not output unconditional information in the c
        target_cond_dict = latent_diffusion.filter_useful_cond_dict(c)

        shape = (latent_diffusion.channels, latent_diffusion.latent_t_size, latent_diffusion.latent_f_size)
        device=latent_diffusion.device
        eta=1.0
        temperature = 1.0
        noise = noise_like(x_init.shape, device, repeat=False) * temperature

        ddim_sampler = DDIMSampler(latent_diffusion, device=device)
        ddim_sampler.make_schedule(ddim_num_steps=ddim_steps, ddim_eta=eta, verbose=False)
        
        timesteps = ddim_sampler.ddim_timesteps

        time_range = np.flip(timesteps)
        total_steps = timesteps.shape[0]
        iterator = tqdm(time_range, desc="DDIM Sampler", total=total_steps, disable=disable_tqdmoutput)
        
        for i, step in enumerate(iterator):
            index = total_steps - i - 1
            t_in = torch.full((batch_size,), step, device=device, dtype=torch.long)

            model_uncond = ddim_sampler.model.apply_model(x_init, t_in, uncond_dict) 
            
            model_target_cond = ddim_sampler.model.apply_model(x_init, t_in, target_cond_dict) 
            
            # CFG; model_output is the estimated error after CFG
            e_t = model_uncond + guidance_scale * (model_target_cond - model_uncond)

            alphas = ddim_sampler.ddim_alphas
            alphas_prev = ddim_sampler.ddim_alphas_prev
    
            sqrt_one_minus_alphas = ddim_sampler.ddim_sqrt_one_minus_alphas
            sigmas = ddim_sampler.ddim_sigmas

            # select parameters corresponding to the currently considered timestep
            a_t = torch.full((batch_size, 1, 1, 1), alphas[index], device=device)
            a_prev = torch.full((batch_size, 1, 1, 1), alphas_prev[index], device=device)
            sigma_t = torch.full((batch_size, 1, 1, 1), sigmas[index], device=device)
            sqrt_one_minus_at = torch.full((batch_size, 1, 1, 1), sqrt_one_minus_alphas[index], device=device)

            noise = sigma_t * noise_like(x_init.shape, device, repeat=False) * temperature
            
            pred_x0 = (x_init - sqrt_one_minus_at * e_t) / a_t.sqrt()
            dir_xt = (1.0 - a_prev - sigma_t**2).sqrt() * e_t
            x_prev = a_prev.sqrt() * pred_x0 + dir_xt + noise

            x_init = x_prev

        mel = latent_diffusion.decode_first_stage(x_prev)
        waveform = latent_diffusion.mel_spectrogram_to_waveform(
            mel, savepath="", bs=None, name="", save=False
        )

        return waveform[0][0]