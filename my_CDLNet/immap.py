import torch 
import torch.fft as fft
import torch.nn as nn
import numpy as np
import h5py
import math
import json
import train
import os
import gc

from mri_utils import batched_mri_encoding, batched_mri_decoding, walsh_smaps, fftc, ifftc, make_acc_mask, quant_complex, quant_tensor, espirit
from functorch import jacrev, jacfwd
from solvers import conj_grad
from pprint import pprint
from functools import partial
from utils import saveimg
from model_utils import uball_project
from metrics import psnr, ssim
from mask import ssdu_mask_subsample

# This code will implement ImMAP: Implicit Maximum a Posteriori estimation for MRI reconstruction
class ImMAP(nn.Module):
    def __init__(self,  denoiser,       # Denoiser to embed image prior
                        beta = 0.05,    # Noise injection ratio, should belong in [0, 1]
                        sigma_L = 0.01, # Noise level cutoff
                        h_0 = 0.01,      # Initial step size
                        lam = 2.,        # Parameter for immap2
                        zeta = 0.5      # Acceleration factor for immap3
                        ):
        super(ImMAP, self).__init__()
        self.denoiser = denoiser
        self.beta = beta
        self.sigma_L = sigma_L
        self.h_0 = h_0
        self.lam = lam
        self.zeta = zeta
    
    def init_diff(self, y, noise_level):
        # Get a random image 
        x_t = torch.randn(1, 1, y.shape[-2], y.shape[-1], dtype = torch.cfloat, device = y.device)
        # Set initial conditions
        t = 1
        sigma_t = torch.Tensor([1.])
        sigma_t = sigma_t.to(y.device)
        sigma_t_prev = sigma_t

        return x_t, t, sigma_t, sigma_t_prev, y

    def forward(self, y, noise_level, acceleration_map, smaps, save_dir = None, verbose = True): # Provide a y to condition on
        # Set initial conditions
        x_t, t, sigma_t, sigma_t_prev, y = self.init_diff(y, noise_level)

        E = partial(batched_mri_encoding, mask = acceleration_map, smaps = smaps)
        EH = partial(batched_mri_decoding, mask = acceleration_map, smaps = smaps)
        
        # Mean shifting
        EHy = EH(y)
        x_t = x_t + torch.mean(EHy)
        with torch.no_grad():
            while sigma_t > self.sigma_L:
                # Get jacobian and denoiser output
                def denoise(x, sigma, f = self.denoiser):
                    x_hat, _ = f(x, sigma)
                    return x_hat
                x_hat_t = denoise(x_t, sigma_t)
                # Get noise level estimate
                sigma_t_sq = torch.mean((x_hat_t - x_t).abs()**2)
                # Tweedie's formula
                grad_prior = x_hat_t - x_t
                # PiGDM Laplace Approx (use * operator because the forward operator E starts with elementwise multiplication
                def S_t(x, noise_level=noise_level, sigma_t_sq = sigma_t_sq, E = E, EH = EH):
                    # We do not actually want to explicitly compute Sigma_t, but rather have the ability to apply it to a matrix
                    x = torch.squeeze(x)
                    return noise_level**2 * x + sigma_t_sq/(1+sigma_t_sq)*E(EH(x))
                # We want to solve sigma_t v_t = E x_hat - y
                # We may use CG since sigma_t is a covariance matrix + PSD symmetric matrix
                v_t, tol_reached = conj_grad(S_t, E(x_hat_t) - y, max_iter = 500, tol=1e-3, verbose = False)
                v_t = torch.squeeze(v_t)
                EHv_t = EH(v_t)
                EHv_t = EHv_t
                # Compute vjp
                _, (grad_likelihood, _) = torch.autograd.functional.vjp(denoise, (x_t, sigma_t), EHv_t)
                grad_likelihood = -1*sigma_t_sq*grad_likelihood
                # Update step size
                h_t = self.h_0 * t/(1+self.h_0*(t-1))
                sigma_t = torch.sqrt(sigma_t_sq)
                # Update noise injection
                gamma_t = sigma_t*((1-self.beta*h_t)**2-(1-h_t)**2)**0.5
                noise = torch.randn_like(x_t)
                # Stochastic gradient ascent
                x_t = x_t + h_t * (grad_prior+grad_likelihood) + gamma_t*noise
                if t % 5 == 0 and save_dir:
                    fname = os.path.join(save_dir, "diffusion_iteration_"+str(t)+".png")
                    saveimg(x_t, fname)
                t = t + 1
                if verbose == True:
                    print(f"Iteration {t} complete. Noise level: {sigma_t}. Tolerance Reached: {tol_reached}") 
                if sigma_t > sigma_t_prev:
                    # Raise flag if noise is greater at next iteration
                    print("Noise is diverging...")
                    continue
                sigma_t_prev = sigma_t
                
            if save_dir:
                fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                saveimg(x_t, fname)
        return x_t

    def forward_2(self, y, noise_level, acceleration_map, smaps, save_dir = None, verbose = True, recon = None, sigma_start = 0.2):
        # Implement ImMAP 2!
        # Set initial conditions
        if recon is None:
            x_t, t, sigma_t, sigma_t_prev, y = self.init_diff(y, noise_level)
        else:
            x_t = recon + sigma_start*torch.randn_like(recon)
            # t = 1 is definitely wrong, we need to find a reasonable t for which we can say sigma_t = 0.2
            t = 120
            sigma_t = sigma_start
        sigma_y = noise_level
        E = partial(batched_mri_encoding, mask = acceleration_map, smaps = smaps)
        EH = partial(batched_mri_decoding, mask = acceleration_map, smaps = smaps)

        # Precompute EHy for calculation
        EHy = EH(y)

        # Fix a noise schedule
        # sig_t_vec = torch.linspace(1.0, 0.001, 50)
        # Mean shifting
        x_t = x_t + torch.mean(EHy)

        with torch.no_grad():
            while sigma_t > self.sigma_L:
                x_hat_t, _ = self.denoiser(x_t, sigma_t)
                # Get noise level estimate
                sigma_t_sq = torch.mean((x_hat_t - x_t).abs()**2)
                sigma_t = torch.sqrt(sigma_t_sq)
                #sigma_t = sig_t_vec[t]
                #sigma_t_sq = sigma_t**2
                # Compute proximal weighting
                p_t = self.lam*sigma_y**2 / (sigma_t_sq/(1+sigma_t_sq))
                # Try inverting rho
                # p_t = (sigma_t_sq/(1+sigma_t_sq)) / (self.lam*sigma_y**2)
                # update step size
                h_t = self.h_0 * t/(1+self.h_0*(t-1))

                # Update noise injection
                gamma_t = sigma_t*((1-self.beta*h_t)**2-(1-h_t)**2)**0.5
                
                # draw random noise
                noise = torch.randn_like(x_t)
                
                # compute proximal update:
                # We want to compute prox_{D/p_t}(x_t)
                # argmin 1/2||y-Ax||^2 + p_t/2||x_t-x||^2
                # derivative is -A^T(y-Ax) - p_t(x_t-x) = 0
                # so, solve for x
                # A^Ty+p_tx_t = (A^TA + p_t*I)x, conjugate gradient here!
                
                def A(x, E = E, EH = EH):
                    return EH(E(x)) + p_t*x
                
                v_t, tol_reached = conj_grad(A, torch.squeeze(p_t*x_hat_t+EHy), max_iter = 500, tol=1e-3, verbose = False)
                # Perform update
                x_t = x_t +h_t*(v_t-x_t) + gamma_t*noise

                if t % 5 == 0 and save_dir:
                    fname = os.path.join(save_dir, "diffusion_iteration_"+str(t)+".png")
                    saveimg(x_t, fname)
                if verbose == True:
                    print(f"Iteration {t} complete. Noise level: {sigma_t}. p_t: {p_t} Tolerance Reached: {tol_reached}")

                t = t + 1
            if save_dir:
                fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                saveimg(x_t, fname)
        return x_t
    def forward_2p5(self,   
                    y, 
                    noise_level, 
                    acceleration_map, 
                    smaps, 
                    e2e_net, 
                    organ_mask, 
                    save_dir = None, 
                    verbose = True, 
                    mode=1, 
                    ssdu_masking = False,
                    ssdu_acs = 10,
                    ssdu_base_accel = 2,
                    ssdu_rho = (0.1, 0.2),
                    acs_lines = 10,
                    ):
        # This implements a version of immap that conditions on an end to end reconstruction using a separate LPDSNet
        # Makes the approximation that E[x|x_t] = e2e_net(x_hat_t, 0, x_t, sigma_t)
                
        # Set initial conditions
        x_t, t, sigma_t, sigma_t_prev, y = self.init_diff(y, noise_level)
        sigma_y = noise_level
        E = partial(batched_mri_encoding, mask = acceleration_map, smaps = smaps)
        EH = partial(batched_mri_decoding, mask = acceleration_map, smaps = smaps)
        # Precompute EHy for calculation
        # Let us fix a noise schedule
        # Precompute the noise schedule first
        # Using Eero's noise schedule
        '''
        sig_t_vec = [1]
        i=1
        
        while sig_t_vec[-1] > self.sigma_L:
            sig_t_vec.append((1-self.beta* self.h_0 * i/(1+self.h_0*(i-1)))*sig_t_vec[i-1])
            i=i+1
        sig_t_vec = torch.tensor(sig_t_vec, device = y.device)
        '''
        EHy = EH(y)
        x_t[:, :, organ_mask] = x_t[:, :, organ_mask] + torch.mean(EHy[:, :, organ_mask])
        # sig_t_vec = torch.linspace(1, self.sigma_L-1e-8, 200, device = y.device)
        with torch.no_grad():
            while sigma_t > self.sigma_L:
                # Network forward pass
                if ssdu_masking is True:
                    # Generate a base mask
                    ssdu_base_mask = make_acc_mask(
                        [smaps.shape[-2], smaps.shape[-1]],
                        ssdu_base_accel,
                        acs_lines
                    )
                    # Cast to bool so we can apply bitwise operations internally
                    ssdu_base_mask = ssdu_base_mask.bool()
                    # Subsample on top of this mask
                    ssdu_mask, _ = ssdu_mask_subsample(
                        ssdu_base_mask,
                        rho = ssdu_rho, # Keep 20% to 80% of lines
                        acs_size = ssdu_acs,
                        type = "uniform_1D"
                    )
                    # Push to GPU
                    ssdu_mask = ssdu_mask.to(y.device)
                    v_t,_ = e2e_net.forward_double_noise(
                        y,
                        0.01,
                        acceleration_map,
                        smaps,
                        x_init=x_t,
                        mri=True,
                        sigma_t=sigma_t,
                        ssdu_mask = ssdu_mask
                    )
                else:
                    v_t, _ = e2e_net.forward_double_noise(y, 0.01, acceleration_map, smaps, x_init = x_t, mri = True, sigma_t = sigma_t)
                v_t = v_t * organ_mask
                # noise level estimation
                sigma_t = torch.sqrt(torch.sum((x_t*organ_mask-v_t*organ_mask).abs()**2)/torch.sum(organ_mask))
                # update step size
                h_t = self.h_0 * t/(1+self.h_0*(t-1))
                # Update noise injection
                gamma_t = sigma_t*((1-self.beta*h_t)**2-(1-h_t)**2)**0.5
                # draw random noise
                noise = torch.randn_like(x_t)
                # Instead of performing a proximal update, use our e2e_net
                # Try stochastic "renoising"
                if t == 1:
                    # grab first iterate
                    first_it = v_t.clone()
                # Replace with ImMAP3 update eqn
                x_t = x_t*organ_mask + h_t * (v_t-x_t) + gamma_t * noise
                # Masking in x_t
                if t % 5 == 0 and save_dir:
                    panel = torch.cat((x_t, v_t), dim = 3)
                    fname = os.path.join(save_dir, "diffusion_noise_level_"+str(round(float(sigma_t.cpu().numpy()), 2))+".png")
                    saveimg(panel, fname, contrast=True)
                if verbose == True:
                    print(f"Iteration {t} complete. Noise level: {sigma_t}.")
                t = t + 1 
            if save_dir:
                fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                saveimg(v_t, fname, contrast=True)
                    
        return x_t, v_t, first_it
    def forward_3(self, y, noise_level, acceleration_map, smaps, save_dir = None, verbose = True):
        # Implments ImMAP 3, basically just DiffPIR
        # Set initial conditions
        x_t, t, _, _, y = self.init_diff(y, noise_level)
        sigma_y = noise_level
        E = partial(batched_mri_encoding, mask = acceleration_map, smaps = smaps)
        EH = partial(batched_mri_decoding, mask = acceleration_map, smaps = smaps)
        # Precompute EHy for calculation
        sig_t_sched = [1]
        i=1
        while sig_t_sched[-1] > 0.01:
            sig_t_sched.append((1-self.beta * self.h_0 * i/(1+self.h_0*(i-1)))*sig_t_sched[i-1])
            i=i+1
        EHy = EH(y)
        sigma_t = 1
        with torch.no_grad():
            while sigma_t > self.sigma_L:
                sigma_t = sig_t_sched[t-1]
                x_hat_t, _ = self.denoiser(x_t, sigma_t)
                p_t = self.lam*sigma_y**2 / (sigma_t**2/(1+sigma_t**2))
                h_t = self.h_0 * t/(1+self.h_0*(t-1))
                gamma_t = sigma_t*h_t*((1-self.beta))**0.5
                noise = torch.randn_like(x_t)
                def A(x, E = E, EH = EH):
                    return EH(E(x)) + p_t*x
                v_t, tol_reached = conj_grad(A, torch.squeeze(p_t*x_hat_t+EHy), max_iter = 1000, tol=1e-3, verbose = False)
                
                x_t = v_t + (1-self.zeta)**0.5 * h_t * (v_t-x_t) + (self.zeta)**0.5 * gamma_t * noise
                if t % 5 == 0 and save_dir:
                    fname = os.path.join(save_dir, "diffusion_iteration_"+str(t)+".png")
                    saveimg(x_t, fname)
                if verbose == True:
                    print(f"Iteration {t} complete. Noise level: {sigma_t}. p_t: {p_t}")
                t = t + 1
            if save_dir:
                fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                saveimg(x_t, fname)
        return x_t
    def forward_3p5(self, y, noise_level, acceleration_map, smaps, e2e_net, save_dir = None, verbose = True):
        # Implments ImMAP 3, basically just DiffPIR
        # Set initial conditions
        x_t, t, _, _, y = self.init_diff(y, noise_level)
        sigma_y = noise_level
        E = partial(batched_mri_encoding, mask = acceleration_map, smaps = smaps)
        EH = partial(batched_mri_decoding, mask = acceleration_map, smaps = smaps)
        # Precompute EHy for calculation
        EHy = EH(y)
        
        sigma_t = 1
        sig_t_sched = torch.linspace(1, 0.001, 100)
        with torch.no_grad():
            while sigma_t > self.sigma_L:
                sigma_t = sig_t_sched[t-1]
                x_hat_t, _ = self.denoiser(x_t, sigma_t)
                p_t = self.lam*sigma_y**2 / (sigma_t**2/(1+sigma_t**2))
                h_t = self.h_0 * t/(1+self.h_0*(t-1))
                gamma_t = sigma_t*h_t*((1-self.beta))**0.5
                noise = torch.randn_like(x_t)
                '''
                def A(x, E = E, EH = EH):
                    return EH(E(x)) + p_t*x
                v_t, tol_reached = conj_grad(A, torch.squeeze(p_t*x_hat_t+EHy), max_iter = 1000, tol=1e-3, verbose = False)
                '''
                v_t, _ = e2e_net.forward_double_noise(y, noise_level, mask = acceleration_map, smaps = smaps, x_init = x_t, mri = True, sigma_t = sigma_t)
                x_t = v_t + (1-self.zeta)**0.5 * h_t * (v_t-x_t) + (self.zeta)**0.5 * gamma_t * noise
                if t % 5 == 0 and save_dir:
                    fname = os.path.join(save_dir, "diffusion_iteration_"+str(t)+".png")
                    saveimg(x_t, fname)
                if verbose == True:
                    print(f"Iteration {t} complete. Noise level: {sigma_t}. p_t: {p_t}")
                t = t + 1
            if save_dir:
                fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                saveimg(x_t, fname)
        self.beta=0.05
        return x_t
    def forward_4(  self, 
                    y, 
                    noise_level, 
                    acceleration_map, 
                    smaps, 
                    e2e_net, 
                    organ_mask, 
                    recon=None, 
                    save_dir = None, 
                    verbose = True, 
                    sigma_T=0.1,
                    ssdu_masking = False,
                    ssdu_acs = 10,
                    ssdu_base_accel = 2,
                    ssdu_rho = (0.1, 0.2),
                    acs_lines = 10,
                    ):
        # In immap4, we take in a given reconstruction, add noise, and then proceed. 
        # x_t, t, _, _, y = self.init_diff(y, noise_level)
        t = 1
        sigma_y = noise_level
        E = partial(batched_mri_encoding, mask = acceleration_map, smaps = smaps)
        EH = partial(batched_mri_decoding, mask = acceleration_map, smaps = smaps)
        # Bring up to a reasonable noise level 
        EHy = EH(y)
        if recon is None:
            recon = EHy
        x_t = recon + sigma_T*torch.randn_like(recon)
        # Perform regular immap2.5 iterations
        # sig_t_vec = torch.linspace(sigma_T, self.sigma_L - 1e-8, int((sigma_T-self.sigma_L)*100), device = y.device)
        # Compute Eero schedule:
        '''
        sig_t_vec = [sigma_T]
        i=1
        
        while sig_t_vec[-1] > self.sigma_L:
            sig_t_vec.append((1-self.beta* self.h_0 * i/(1+self.h_0*(i-1)))*sig_t_vec[i-1])
            i=i+1
        sig_t_vec = torch.tensor(sig_t_vec, device = y.device)
        
        t_offset = 0
        '''
        sigma_t = torch.tensor([sigma_T], device = y.device)
        t_offset = 0
        with torch.no_grad():
            while sigma_t > self.sigma_L:
                # Instead of performing a proximal update, use our e2e_net
                if ssdu_masking is True:
                    # Generate a base mask
                    ssdu_base_mask = make_acc_mask(
                        [smaps.shape[-2], smaps.shape[-1]],
                        ssdu_base_accel,
                        acs_lines
                    )
                    # Cast to bool so we can apply bitwise operations internally
                    ssdu_base_mask = ssdu_base_mask.bool()
                    # Subsample on top of this mask
                    ssdu_mask, _ = ssdu_mask_subsample(
                        ssdu_base_mask,
                        rho = ssdu_rho, # Keep 20% to 80% of lines
                        acs_size = ssdu_acs,
                        type = "uniform_1D"
                    )
                    # Push to GPU
                    ssdu_mask = ssdu_mask.to(y.device)
                    v_t,_ = e2e_net.forward_double_noise(
                        y,
                        noise_level,
                        acceleration_map,
                        smaps,
                        x_init=x_t,
                        mri=True,
                        sigma_t=sigma_t,
                        ssdu_mask = ssdu_mask
                    )
                else:
                    v_t, _ = e2e_net.forward_double_noise(y, noise_level, acceleration_map, smaps, x_init = x_t, mri = True, sigma_t = sigma_t)
                v_t = v_t*organ_mask
                # x_hat_t, _ = self.denoiser(x_t, sigma_t)
                # sigma_t = torch.sqrt(torch.mean((recon - x_t).abs()**2))
                sigma_t = torch.sqrt(torch.sum((x_t*organ_mask-v_t*organ_mask).abs()**2)/torch.sum(organ_mask))
                # sigma_t = sig_t_vec[t-1]
                # update step size
                h_t = self.h_0 * (t+t_offset)/(1+self.h_0*(t+t_offset-1))
                # Update noise injection
                gamma_t = sigma_t*((1-self.beta*h_t)**2-(1-h_t)**2)**0.5
                # draw random noise
                noise = torch.randn_like(x_t)

                if t == 1:
                    # grab first iterate
                    first_it = v_t.clone()
                # Replace with ImMAP3 update eqn
                x_t = x_t*organ_mask + h_t * (v_t-x_t) + gamma_t * noise
                # x_t = v_t + sigma_t * torch.randn_like(v_t) + h_t * (v_t-x_t) + gamma_t * noise
                if t % 10 == 0 and save_dir:
                    panel = torch.cat((x_t, v_t, recon), dim = 3)
                    fname = os.path.join(save_dir, "diffusion_noise_level_"+str(round(float(sigma_t.cpu().numpy()), 2))+".png")
                    saveimg(panel, fname, contrast=True)
                if verbose == True:
                    print(f"Iteration {t} complete. Noise level: {sigma_t}.")
                t = t + 1
            if save_dir:
                fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                saveimg(v_t, fname, contrast=True)
        return v_t

