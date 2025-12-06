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

from mri_utils import mri_encoding, mri_decoding, walsh_smaps, fftc, ifftc, make_acc_mask
from functorch import jacrev, jacfwd
from solvers import conj_grad
from pprint import pprint
from functools import partial
from utils import saveimg

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("args_fn", type=str, help="Path to args.json file.", default="args.json")
parser.add_argument("--kspace_path", type = str, help="Corresponding path where kspace data can be found", default = None)
parser.add_argument("--noise_level", type = float, help="Std deviation of injected noise into kspace data", default = 0.)

args = parser.parse_args()

# This code will implement ImMAP: Implicit Maximum a Posteriori estimation for MRI reconstruction
class ImMAP(nn.Module):
    def __init__(self,  denoiser,       # Denoiser to embed image prior
                        beta = 0.05,    # Noise injection ratio, should belong in [0, 1]
                        sigma_L = 0.01, # Noise level cutoff
                        h_0 = 0.01      # Initial step size
                        ):
        super(ImMAP, self).__init__()
        self.denoiser = denoiser
        self.beta = beta
        self.sigma_L = sigma_L
        self.h_0 = h_0
    
    def forward(self, y, noise_level, acceleration_map, smaps): # Provide a y to condition on
        # Get a random image 
        x_t = torch.randn(y.shape[-2], y.shape[-1], dtype = torch.cfloat, device = y.device)
        x_t = x_t[None, None, :, :]
        # Set initial conditions
        t = 1
        sigma_t = torch.Tensor([1.])
        sigma_t = sigma_t.to(y.device)
        sigma_t_prev = sigma_t
        E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
        EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)
        # Add noise to y
        y = y + noise_level * torch.randn_like(y)
        with torch.no_grad():
            while sigma_t > self.sigma_L:
                # Get jacobian and denoiser output
                def denoise(x, sigma, f = self.denoiser):
                    x_hat, _ = f(x, sigma*255.)
                    return x_hat
                x_hat_t = denoise(x_t, sigma_t)
                # Get noise level estimate
                sigma_t_sq = torch.mean((x_hat_t - x_t).abs()**2)
                sigma_t = torch.sqrt(sigma_t_sq)
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
                # Update noise injection
                gamma_t = sigma_t*((1-self.beta*h_t)**2-(1-h_t)**2)**0.5
                noise = torch.randn_like(x_t)
                # Stochastic gradient ascent
                x_t = x_t + h_t * (grad_prior+grad_likelihood) + gamma_t*noise
                if t % 5 == 0:
                    fname = os.path.join("diff_figs", "diffusion_iteration_"+str(t)+".png")
                    saveimg(x_t, fname)
                t = t + 1
                print(f"Iteration {t} complete. Noise level: {sigma_t}") 
                sigma_t_prev = sigma_t
        return x_t

def main(args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    slice = 3

    kspace_fname = args.kspace_path
    fname = os.path.basename(kspace_fname)

    # Search in val dir for corresponding smaps
    smaps_fname = os.path.join("../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val", fname)
    
    with h5py.File(smaps_fname) as f:
        smaps = f['smaps'][:, :, :, :]
        smaps = smaps[slice, :, :, :]
        gnd_truth = f['image'][slice, :, :]

    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :, :]
    # Squeeze smaps, also conjugate since they come as conjugated form
    # smaps = smaps[0, :, :, :].conj()
    kspace = torch.from_numpy(kspace)
    # breakpoint()
    # smaps = walsh_smaps(ifftc(kspace[None]))
    smaps = torch.from_numpy(smaps)
    smaps = torch.squeeze(smaps)
    # Detect acceleration maps
    #mask = detect_acc_mask(kspace)
    
    _, mask = make_acc_mask(shape = (smaps.shape[1], smaps.shape[2]), accel = 8, acs_lines = 24)
    # Send to GPU
    smaps = smaps.to(device)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*2e3
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = mask * kspace
    # Get noise level 
    noise_level = args.noise_level

    # Load CDLNet denoiser
    model_args_file = open(args.args_fn)
    model_args = json.load(model_args_file)
    pprint(model_args)
    
    # Verify kspace and smap stuff works
    
    # Check EHy
    # breakpoint()
    # EHy = mri_decoding(kspace, torch.ones(smaps.shape[1], smaps.shape[2], device = smaps.device), smaps)
    # saveimg(EHy, "Ehy.png")
    # breakpoint()
    net, _, _, _ = train.init_model(model_args, device=device, quant_ckpt = True)
    net.eval()
    immap = ImMAP(net)
    test = immap(kspace_masked, noise_level, mask, smaps)
    breakpoint()

if __name__ == "__main__":
    main(args)
