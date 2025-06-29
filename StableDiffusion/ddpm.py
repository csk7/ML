from inspect import getgeneratorlocals
from numpy import linspace
import torch
from torch.nn import nn
import torch.nn.functional as F

class ddpm_sampler(nn.Module):
    def __init__(self, generator = torch.Generator, num_training_steps:int=1000, beta_start = 0.00085, beta_end = 0.0120):
        self.betas = linspace(beta_start ** 0.5, beta_end ** 0.5, num_training_steps, dtype = torch.float32) ** 0.5
        self.alphas = 1 - self.betas
        self.alpha_cum_prod = torch.cumprod(self.alphas)
        self.one = torch.tensor(1.0)

        self.generator = generator
        self.num_training_steps = num_training_steps
        self.timesteps = torch.arange(0, self.num_training_steps-1).flip(dims = [0])

    def set_inference_steps(self, num_inference_steps:int = 50):
        self.num_inference_steps = num_inference_steps
        step_ratio = self.num_training_steps//self.num_inference_steps
        self.timesteps = torch.arange(0, self.num_training_steps-1, step_ratio, dtype=torch.int64).flip(dims = [0])

    def _get_prev_timestep(self, init_timstep):
        return init_timstep - self.num_training_steps//self.num_inference_steps

    def _get_variance(self, init_timstep):
        t = timestep
        prev_t = _get_prev_timestep(timestep)
        alpha_bar_t = self.alpha_cum_prod[timestep]
        prev_alpha_bar_t = self.alpha_cum_prod[prev_t] if prev_t >=0 else self.one
        beta_t = 1 - alpha_bar_t
        prev_beta_t = 1 - prev_alpha_bar_t
        current_alpha_t = alpha_bar_t / prev_alpha_bar_t
        current_beta_t = 1 - current_alpha_t

        variance = (prev_beta_t / beta_t) * current_beta_t

        variance = torch.clamp(variance, min = 1e-20)
        return variance
        

    def step(self, timestep:int, latents:torch.FloatTensor, model_output:torch.FloatTensor):
        '''
        latents, model_output: [B, 4, H/8, W/8]
        '''
        t = timestep
        prev_t = _get_prev_timestep(timestep)
        alpha_bar_t = self.alpha_cum_prod[timestep]
        prev_alpha_bar_t = self.alpha_cum_prod[prev_t] if prev_t >=0 else self.one
        beta_t = 1 - alpha_bar_t
        prev_beta_t = 1 - prev_alpha_bar_t
        current_alpha_t = alpha_bar_t / prev_alpha_bar_t
        current_beta_t = 1 - current_alpha_t

        #DDPM Formuale 15
        predicted_x0 = (latents - (beta_t ** 0.5)*model_output)/(alpha_bar_t ** 0.5)

        #Compute coefffiecient in eq 7
        x0_coeff = ((prev_alpha_bar_t ** 0.5) * current_beta_t)/beta_t
        xt_xoeff = (current_alpha_t ** 0.5)*prev_beta_t/(beta_t)

        #Compute predicte prev mean and variance
        mean_prev_x = x0_coeff * predicted_x0 + xt_xoeff * latents

        variance_prev_x = 0
        if(t>0):
            noise = torch.randn((model_output.shape), generator=self.generator, device=model_output.device, dtype=model_output.dtype)
            variance_prev_x = _get_variance(self, init_timstep)
        
        #N(0,1) --> N(mean_prev, var_prev)
        prev_distribution = mean_prev_x + noise * (variance_prev_x ** 0.5)
        return prev_distribution

    def set_strength(self, strength: int):
        '''
        we need to add noise to the latent image in the image to image model, from where sampler will start
        We do that by altering the steps, if strength is more, we skip more steps. Fooling sampler that it came up with htis noise level
        '''
        start_steps = self.num_inference_steps - int(strength*self.num_inference_steps)
        self.timesteps = self.timesteps[start_steps:]
        self.start_step = start_steps

    def add_noise(self, original_sample:torch.FloatTensor, timesteps:torch.IntTensor) -> torch.FloatTensor:
        '''
        original_sample: first least noisy image (B,C,H,W)
        timesteps: timesteps to add noise from first image
        '''
        alpa_bar = self.alpha_cum_prod.to(device = original_samples.device, dtype = original_samples.dtype)
        timesteps = timesteps.to(device = original_sample.device)

        #alpha has all trianing time steps, we need only inference ones that are updated, so taking a subset
        mean_noise_added = apha_bar[timesteps] ** 0.5
        mean_noise_added = mean_noise_added.flatten()

        while(len(mean_noise_added.shape)<len(original_sample.shape)):
            mean_noise_added = mean_noise_added.unsqueeze(-1)

        std_dev_noise_added = (1-alpa_bar[timesteps]) ** 0.5
        std_dev_noise_added = std_noise_added.flatten
        while(len(std_noise_added.shape)<len(original_sample.shape)):
            std_dev_noise_added = std_dev_noise_added.unsqueeze(-1)

        #Fwd noise adding steps
        noise = torch.randn((original_sample.shape), generator=self.generator, device=original_samples.device, dtype=original_samples.dtype)
        noise_image = noise*std_dev_noise_added + mean_noise_added*original_sample
        return noise_image #For all time steps

