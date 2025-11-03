import torch 
import torch.fft as fft
import numpy as np
import h5py
import math
from torch.func import jacrev

# This code will implement ImMAP: Implicit Maximum a Posteriori estimation for MRI reconstruction

# Let us write a helper function to give us a DFT matrix for some N
def dft_matrix(N):
    # Generate a vector [0, 1, ..., N-1]
    n = torch.linspace(0, N-1, N)
    # Generate meshgrid based on n
    Nx, Ny = torch.meshgrid(n, n)
    W = Nx * Ny
    # Turn our W matrix into a complex valued matrix with W as the imaginary components
    W = torch.complex(torch.zeros_like(W), -2*torch.Tensor([math.pi])*W/N)
    return torch.exp(W)

def mri_encoding(acceleration_map, smap):
    # We take an acceleration_map to be a row-removed identity matrix corresponding to how many lines in kspace we keep
    # We take a sensitivity map and assume it performs elementwise multiplication in the image domain
    # Return an MRI encoding matrix!
    # Assume smap is square 
    N = smap.shape[-1]
    dft_mat = dft_matrix(N)
    out = acceleration_map @ dft_mat @ smap
    return out

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
    
    def forward(self, y, sigma_y, acceleration_map, smap): # Provide a y to condition on
        # Get a random image 
        x_0 = torch.randn_like(y)
        # Set initial conditions
        t = 1
        sigma_t = 1
        E = mri_encoding(acceleration_map, smap)
        E_H = torch.adjoint(E)
        while sigma_t > sigma_L:
            # Get jacobian and denoiser output
            J_t, x_hat_t = jacrev(self.denoiser, has_aux = True)(x_0, sigma_t)
            # Get noise level estimate
            sigma_t = torch.sqrt(torch.mean((x_hat_t - x_t).abs()**2))
            # Tweedie's formula
            grad_prior = x_hat_t - x_t
            # PiGDM Laplace Approx (use * operator because the forward operator E starts with elementwise multiplication
            sigma_t = sigma_y + sigma_t**2/(1+sigma_t**2)*E*E_H
            v_t = torch.linalg.inv(sigma_t)@(E*x_hat-y)
            grad_likelihood = -J_t.adjoint()@ E_H@y
            # Update step size
            h_t = self.h_0 * (t/1+h_0*(t-1))
            # Update noise injection
            gamma_t = sigma_t*torch.sqrt((1-self.beta*h_t)**2-(1-h_t)**2)
            noise = torch.randn_like(x_t)
            # Stochastic gradient ascent
            x_t = x_t + h_t * (grad_prior+grad_likelihood) + gamma_t*noise
            t = t + 1
        return x_t
