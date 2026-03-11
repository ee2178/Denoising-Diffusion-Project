import torch
import torch.fft as fft
import torch.nn as nn
import numpy as np

from mri_utils import mri_encoding, mri_decoding, walsh_smaps, fftc, ifftc, make_acc_mask, quant_complex, quant_tensor, espirit
from solvers import conj_grad
from pprint import pprint
from functools import partial
from utils import saveimg



# This code will implement ImMAP: Implicit Maximum a Posteriori estimation for MRI reconstruction
class DDS(nn.Module):
    def __init__(self,  denoiser,       # Denoiser to embed image prior
                        beta = 0.05,    # Noise injection ratio, should belong in [0, 1]
                        sigma_L = 0.01, # Noise level cutoff
                        h_0 = 0.01,     # Initial step size
                        eta = 0.8,      # stochasticity of ddim update
                        gamma = 1,      # data consistency weight
                        ):
        super(DDS, self).__init__()
        self.denoiser = denoiser
        self.eta = eta
        self.gamma = gamma
        self.beta = beta
        self.sigma_L = sigma_L
        self.h_0 = h_0

    def prep_noise_schedule(self, mode = 'linear', sigma_start = 1.):
        # Return a noise schedule based on inputs.
        if mode == 'linear':
            ns = torch.linspace(sigma_start, 0, 101)
        if mode == 'eero':
            ns = [sigma_start]
            i=1
            while ns[-1] > 0.01:
                ns.append((1-self.beta * self.h_0 * i/(1+self.h_0*(i-1)))*ns[i-1])
                i=i+1
            ns = torch.tensor(ns)
        return ns

    def forward(self, y, noise_level, acceleration_map, smaps, save_dir = None, verbose = True, sched = 'linear', maxit = 1e4, tol=1e-3):
        # DDS Forward pass, assume y is noisy kspace with sigma_y = noise_level
        # Assume no warm start
        sig_t_vec = self.prep_noise_schedule(mode = sched)

        # Define MRI operators
        E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
        EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)
        
        # Compute alphas from sigmas
        alpha_bar_vec = 1/(1+sig_t_vec**2)
        alpha_vec = torch.zeros_like(alpha_bar_vec)
        alpha_vec[0] = alpha_bar_vec[0] # First elements are the same, following elements are ratios of consecutive entries
        alpha_vec[1:] = alpha_bar_vec[1:]/alpha_vec[:-1]

        # Initialize initial x
        x_t = torch.randn(y.shape[-2], y.shape[-1], dtype = torch.cfloat, device = y.device)
        x_t = x_t[None, None, :, :]

        # Preselect proximal weight, let's forget any type of lambda scaling
        rho = self.gamma

        # Precompute EHy for calculation
        EHy = EH(y)
        t = 0
        sigma_t = sig_t_vec[t]
        with torch.no_grad():
        #    while sigma_t >= self.sigma_L:
            for t in range(len(sig_t_vec)-1):               
                # Compute denoised estimate
                x_hat_t, _ = self.denoiser(x_t, sigma_t*255.)
                # Get noise level estimate
                sigma_t = sig_t_vec[t]
                # Estimate error
                eps_t = (x_t-alpha_bar_vec[t]**(1/2)*x_hat_t)/(1-alpha_bar_vec[t]**(1/2))
                # draw random noise
                noise = torch.randn_like(x_t)
                def A(x, E = E, EH = EH):
                    return rho*EH(E(x)) + x

                xp_t, tol_reached = conj_grad(A, torch.squeeze(x_hat_t+rho*EHy), max_iter = maxit, tol=tol, verbose = False)

                # Perform update
                beta_t = 1 - alpha_vec[t]
                r = alpha_bar_vec[t]/alpha_bar_vec[t+1]
                beta_bar_t = ((1-alpha_bar_vec[t+1])/(1-alpha_bar_vec[t]))**(1/2)*(1-r)
                sigma_bar_t = self.eta * beta_bar_t

                x_t = alpha_bar_vec[t+1]**(1/2)*xp_t - (1-alpha_bar_vec[t+1]-sigma_bar_t**2)**(1/2)*eps_t + sigma_bar_t * noise

                if t % 5 == 0 and save_dir:
                    fname = os.path.join(save_dir, "diffusion_iteration_"+str(t)+".png")
                    saveimg(x_t, fname)
                if verbose == True:
                    print(f"Iteration {t} complete. Noise level: {sigma_t}.")

                t = t + 1
            if save_dir:
                fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                saveimg(x_t, fname)
        # Final tweedie correction
        # We know, now, sigma_t = sigma_L. So take one final step without data consistency 
        x_hat_t, _ = self.denoiser(x_t, sig_t_vec[-1]*255.)
        x_t = x_t + sig_t_vec[-1]**2*(x_hat_t-x_t)
        return x_t
