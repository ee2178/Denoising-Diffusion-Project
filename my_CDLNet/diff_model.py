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

from mri_utils import mri_encoding, mri_decoding, walsh_smaps, fftc, ifftc, make_acc_mask, quant_complex, quant_tensor
from functorch import jacrev, jacfwd
from solvers import conj_grad
from pprint import pprint
from functools import partial
from utils import saveimg
from model_utils import uball_project

'''import argparse
parser = argparse.ArgumentParser()
parser.add_argument("args_fn", type=str, help="Path to args.json file.", default="args.json")
parser.add_argument("--kspace_path", type = str, help="Corresponding path where kspace data can be found", default = None)
parser.add_argument("--noise_level", type = float, help="Std deviation of injected noise into kspace data", default = 0.)
parser.add_argument("--save_dir", type=str, help="Directory to save iterations to", default = None)

args = parser.parse_args()'''

# This code will implement ImMAP: Implicit Maximum a Posteriori estimation for MRI reconstruction
class ImMAP(nn.Module):
    def __init__(self,  denoiser,       # Denoiser to embed image prior
                        beta = 0.05,    # Noise injection ratio, should belong in [0, 1]
                        sigma_L = 0.01, # Noise level cutoff
                        h_0 = 0.01,      # Initial step size
                        lam = 2.        # Parameter for immap2
                        ):
        super(ImMAP, self).__init__()
        self.denoiser = denoiser
        self.beta = beta
        self.sigma_L = sigma_L
        self.h_0 = h_0
        self.lam = lam
    
    def init_diff(self, y, noise_level):
        # Get a random image 
        x_t = torch.randn(y.shape[-2], y.shape[-1], dtype = torch.cfloat, device = y.device)
        x_t = x_t[None, None, :, :]
        # Set initial conditions
        t = 1
        sigma_t = torch.Tensor([1.])
        sigma_t = sigma_t.to(y.device)
        sigma_t_prev = sigma_t
        # Add noise to y
        noisy_y = y + noise_level * torch.randn_like(y)

        return x_t, t, sigma_t, sigma_t_prev, noisy_y

    def forward(self, y, noise_level, acceleration_map, smaps, save_dir = None, verbose = False): # Provide a y to condition on
        # Set initial conditions
        x_t, t, sigma_t, sigma_t_prev, y = self.init_diff(y, noise_level)

        E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
        EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)
        with torch.no_grad():
            while sigma_t > self.sigma_L:
                # Get jacobian and denoiser output
                def denoise(x, sigma, f = self.denoiser):
                    x_hat, _ = f(x, sigma*255.)
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
                v_t, tol_reached = conj_grad(S_t, E(x_hat_t) - y, max_iter = 1e5, tol=1e-2, verbose = False)
                v_t = torch.squeeze(v_t)
                EHv_t = EH(v_t)
                EHv_t = EHv_t[None, None, :, :]
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
                    print(f"Iteration {t} complete. Noise level: {sigma_t}") 
                if sigma_t > sigma_t_prev:
                    # Raise flag if noise is greater at next iteration
                    print("Noise is diverging...")
                    continue
                sigma_t_prev = sigma_t
                
            if save_dir:
                fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                saveimg(x_t, fname)
        return x_t

    def forward_2(self, y, noise_level, acceleration_map, smaps, save_dir = None, verbose = True):
        # Implement ImMAP 2!
        # Set initial conditions
        x_t, t, sigma_t, sigma_t_prev, y = self.init_diff(y, noise_level)
        sigma_y = noise_level
        E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
        EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)

        # Precompute EHy for calculation
        EHy = EH(y)

        with torch.no_grad():
            while sigma_t > self.sigma_L:
                x_hat_t, _ = self.denoiser(x_t, sigma_t*255.)
                # Get noise level estimate
                sigma_t_sq = torch.mean((x_hat_t - x_t).abs()**2)
                sigma_t = torch.sqrt(sigma_t_sq)

                # Compute proximal weighting
                p_t = self.lam*sigma_y**2 / (sigma_t_sq/(1+sigma_t_sq))
               
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
                
                prox_update, tol_reached = conj_grad(A, torch.squeeze(p_t*x_hat_t+EHy), max_iter = 100, tol=1e-3, verbose = False)
                
                '''
                # Use derived result for prox of l2 norm - this doesn't work!!!!!
                prox_update = torch.maximum(torch.zeros_like(x_hat_t).real, 1-1/(p_t*x_hat_t.abs()))*x_hat_t
                '''
                # Perform update
                x_t = x_t + h_t * (prox_update-x_t) + gamma_t*noise
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
    def forward_quant_smaps(self, y, noise_level, acceleration_map, smaps, save_dir = None, mode = 2, n_bits = 4):
        # A method to experiment with different quantization steps within diffusion process
        # We will focus mostly on quantization of smaps
        
        smaps_quant = quant_complex(smaps, n_bits, mag_quant = False, clipping_factor = 1.0)
        # smaps_quant = smaps
        if mode == 2:
            out = self.forward_2(y, noise_level, acceleration_map, smaps, save_dir)
        if mode == 1:
            out = self.forward(y, noise_level, acceleration_map, smaps, save_dir)
        return out
    def forward_quant(self, y, noise_level, acceleration_map, smaps, save_dir = None, verbose = True, mode = 2, n_bits = 16, clipping_factor = 1.0):
        # Set initial conditions
        x_t, t, sigma_t, sigma_t_prev, y = self.init_diff(y, noise_level)
        
        # Try quantizing smaps down to 4 bit!
        # smaps = quant_complex(smaps, 4, mag_quant = False, clipping_factor = 1.0)

        E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
        EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)
        with torch.no_grad():
            while sigma_t > self.sigma_L:
                # Get jacobian and denoiser output
                def denoise(x, sigma, f = self.denoiser, n_bits = n_bits, clipping_factor = clipping_factor):
                    x_hat, _ = f.forward_quant(x, sigma*255., n_bits = n_bits, clipping_factor = clipping_factor)
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
                v_t, tol_reached = conj_grad(S_t, E(x_hat_t) - y, max_iter = 1e5, tol=1e-2, verbose = False)
                v_t = torch.squeeze(v_t)
                EHv_t = EH(v_t)
                EHv_t = EHv_t[None, None, :, :]
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
                    print(f"Iteration {t} complete. Noise level: {sigma_t}")
                if sigma_t > sigma_t_prev:
                    # Raise flag if noise is greater at next iteration
                    print("Noise is diverging...")
                    continue
                sigma_t_prev = sigma_t
            if save_dir:
                fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                saveimg(x_t, fname)
        return x_t

    def forward_2_e2econditioned(self, y, noise_level, acceleration_map, smaps, e2e_net, save_dir = None, verbose = True, mode=1):
        # This implements a version of immap that conditions on an end to end reconstruction using a separate LPDSNet
        # Makes the approximation that E[x|x_t] = e2e_net(x_hat_t, 0, x_t, sigma_t)
                
        # Set initial conditions
        x_t, t, sigma_t, sigma_t_prev, y = self.init_diff(y, noise_level)
        sigma_y = noise_level
        E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
        EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)
        # Precompute EHy for calculation
        EHy = EH(y)
        if mode == 1:
            with torch.no_grad():
                while sigma_t > self.sigma_L:
                    x_hat_t, _ = self.denoiser(x_t, sigma_t*255.)
                    # Get noise level estimate
                    sigma_t_sq = torch.mean((x_hat_t - x_t).abs()**2)
                    sigma_t = torch.sqrt(sigma_t_sq)
                    # update step size
                    h_t = self.h_0 * t/(1+self.h_0*(t-1))
                    # Update noise injection
                    gamma_t = sigma_t*((1-self.beta*h_t)**2-(1-h_t)**2)**0.5
                    # draw random noise
                    noise = torch.randn_like(x_t)
                    # Instead of performing a proximal update, use our e2e_net
                    breakpoint()
                    x_p, _ = e2e_net(y[None], noise_level*255., mask = acceleration_map[None], smaps = smaps[None], x_init = x_hat_t, mri = True)
                    x_t = x_t + h_t * (x_p-x_t) + gamma_t*noise
                    if t % 5 == 0 and save_dir:
                        fname = os.path.join(save_dir, "diffusion_iteration_"+str(t)+".png")
                        saveimg(x_t, fname)
                    if verbose == True:
                        print(f"Iteration {t} complete. Noise level: {sigma_t}.")
                    t = t + 1 
                if save_dir:
                    fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                    saveimg(x_t, fname)
        if mode == 2:
            # This version only does the e2e conditioning for a single step at the start
            with torch.no_grad():
                x_hat_t, _ = self.denoiser(x_t, sigma_t*255.)
                # Get noise level estimate
                sigma_t_sq = torch.mean((x_hat_t - x_t).abs()**2)
                sigma_t = torch.sqrt(sigma_t_sq)
                # update step size
                h_t = self.h_0 * t/(1+self.h_0*(t-1))
                # Update noise injection
                gamma_t = sigma_t*((1-self.beta*h_t)**2-(1-h_t)**2)**0.5
                # draw random noise
                noise = torch.randn_like(x_t)
                # Instead of performing a proximal update, use our e2e_net
                x_p = e2e_net(y, noise_level, x_init = x_hat_t, mri = True)
                x_t = x_t + h_t * (x_p-x_t) + gamma_t*noise
                # Then, proceed with just regular immap
                while sigma_t > self.sigma_L:
                    x_hat_t, _ = self.denoiser(x_t, sigma_t*255.)
                    # Get noise level estimate
                    sigma_t_sq = torch.mean((x_hat_t - x_t).abs()**2)
                    sigma_t = torch.sqrt(sigma_t_sq)
                    # update step size
                    h_t = self.h_0 * t/(1+self.h_0*(t-1))
                    # Update noise injection
                    gamma_t = sigma_t*((1-self.beta*h_t)**2-(1-h_t)**2)**0.5
                    # draw random noise
                    noise = torch.randn_like(x_t)
                    # Instead of performing a proximal update, use our e2e_net
                    x_p = e2e_net(x_t, sigma_t, x_init = x_hat_t, mri = True)
                    x_t = x_t + h_t * (x_p-x_t) + gamma_t*noise
                    if t % 5 == 0 and save_dir:
                        fname = os.path.join(save_dir, "diffusion_iteration_"+str(t)+".png")
                        saveimg(x_t, fname)
                    if verbose == True:
                        print(f"Iteration {t} complete. Noise level: {sigma_t}. p_t: {p_t}")
                    t = t + 1
                if save_dir:
                    fname = os.path.join(save_dir, "diffusion_iteration_"+str(t-1)+".png")
                    saveimg(x_t, fname)
    def forward_3(self, y, noise_level, acceleration_map, smaps, e2e_net, save_dir = None, verbose = True, sig_t_sched = torch.linspace(1, 0, 50), zeta = 0.5):
        # Implments ImMAP 3, basically just DiffPIR
        # Set initial conditions
        x_t, t, _, _, y = self.init_diff(y, noise_level)
        sigma_y = noise_level
        E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
        EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)
        # Precompute EHy for calculation
        EHy = EH(y)
        with torch.no_grad():
            while sigma_t > self.sigma_L:
                sigma_t = sig_t_sched[t]
                x_t, _ = self.denoiser(x_t, sigma_t)
                p_t = self.lam*sigma_y**2 / (sigma_t_sq/(1+sigma_t**2))
                h_t = self.h_0 * t/(1+self.h_0*(t-1))
                gamma_t = sigma_t*((1-self.beta*h_t)**2-(1-h_t)**2)**0.5
                noise = torch.randn_like(x_t)
                def A(x, E = E, EH = EH):
                    return EH(E(x)) + p_t*x
                v_t, tol_reached = conj_grad(A, torch.squeeze(p_t*x_hat_t+EHy), max_iter = 1000, tol=1e-3, verbose = False)
                
                x_t = v_t + torch.sqrt(1-zeta) * h_t * (prox_update-x_t) + torch.sqrt(zeta) * gamma_t * noise
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

def main():
    # test on one specific sample
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    slice = 3

    kspace_fname = "../../datasets/fastmri/brain/multicoil_val/file_brain_AXT2_200_2000572.h5"
    fname = os.path.basename(kspace_fname)

    # Search in val dir for corresponding smaps
    smaps_fname = os.path.join("../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val", fname)
    
    with h5py.File(smaps_fname) as f:
        smaps = f['smaps'][:, :, :, :]
        smaps = smaps[slice, :, :, :]
        gnd_truth = f['image'][slice, :, :]

    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :, :]
    kspace = torch.from_numpy(kspace)
    # breakpoint()
    smaps = torch.from_numpy(smaps)
    smaps = torch.squeeze(smaps)
    
    _, mask = make_acc_mask(shape = (smaps.shape[1], smaps.shape[2]), accel = 8, acs_lines = 24)
    # Send to GPU
    smaps = smaps.to(device)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*2e3
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = mask * kspace
    # Get noise level 
    noise_level = 0.0
    
    gnd_truth = torch.from_numpy(gnd_truth).to(device)*2e3

    # Load CDLNet denoiser
    model_args_file = open("eval_config.json")
    model_args = json.load(model_args_file)
    model_args_file.close()
    pprint(model_args)
    
    save_dir = "diff_figs"

    net, _, _, _ = train.init_model(model_args, device=device, quant_ckpt = True)
    net.eval()
    

    # Load LPDSNet
    lpds_args_file = open("mri_config.json")
    lpds_args = json.load(lpds_args_file)
    lpds_args_file.close()

    lpdsnet, _, _, _ = train.init_model(lpds_args, device = device)
    # Make a noisy kspace measurement
    # noisy_kspace = kspace_masked + noise_level*torch.randn_like(kspace_masked)
    # e2e_recon, _ = lpdsnet(noisy_kspace[None], noise_level*255., mask = mask[None], smaps = smaps[None], mri = True)

    immap = ImMAP(net)
    # test = immap.forward_quant(kspace_masked, noise_level, mask, smaps, save_dir)
    test = immap.forward_2_e2econditioned(kspace_masked, noise_level, mask, smaps, lpdsnet, save_dir = None, verbose = True, mode=1)
    breakpoint()

if __name__ == "__main__":
    main()
