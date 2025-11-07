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

def multicoil_dft_matrix(N, C)
    dft_mat = dft_matrix(N)
    # Generate a block diagonal matrix with C repeats
    # We will do this in a suboptimal way since this only has to be computed once lol
    multicoil_dft_mat = dft_mat
    for i in range(C-1):
        multicoil_dft_mat = torch.block_diag(dft_mat, multicoil_dft_mat)
    return multicoil_dft_mat

def mri_encoding(   acceleration_map,   # sqrt(N) x sqrt(N)
                    smap                # C x sqrt(N) x sqrt(N)
                    ):
    # We take an acceleration_map to be a row-removed identity matrix corresponding to how many lines in kspace we keep
    # We take a sensitivity map and assume it performs elementwise multiplication in the image domain
    # Return an MRI encoding matrix!
    # Assume smap is square in last two dimensions
    N = smap.shape[-1]**2
    C = smap.shape[0]
    # E = MFR
    # R is the sensitivity map operator
    R = torch.diag(smap[0, :, :])
    # M is the Fourier subsampling operator, duplicated C times in block diag fashion
    M = acceleration_map
    for i in range(C-1):
        R = torch.block_diag(R, torch.diag(smap[i, :, :]))
        M = torch.block_diag(M, acceleration_map)
    # F is the N dimensional fourier transform matrix duplicated C times
    F = multicoil_dft_matrix(N, C)
    # E is the overall MRI encoding operator in the multicoil case, a CN x N matrix
    E = M@F@R
    return E

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
        # Assume y is C x sqrt(N) x sqrt(N)
        C = smap.shape[0]
        N = smap.shape[1]**2
        # Get x_0 as sqrt(N) x sqrt(N)
        x_0 = torch.randn_like(smap[0, :, :])
        # Turn y into C x N
        y = y.view(C, -1)
        # Turn y into CN x 1
        y = y.view(C*N, 1)
        # Set initial conditions
        t = 1
        sigma_t = 1
        # E \in CN x N
        E = mri_encoding(acceleration_map, smap)
        # E^H \in N x CN
        E_H = torch.adjoint(E)
        while sigma_t > sigma_L:
            # Get jacobian and denoiser output, Jacobian N x N, x_hat in img form
            J_t, x_hat_t = jacrev(self.denoiser, has_aux = True)(x_0, sigma_t)
            # Get noise level estimate
            sigma_t = torch.sqrt(torch.mean((x_hat_t - x_t).abs()**2))
            # Tweedie's formula (grad prior sqrtN x sqrtN)
            grad_prior = x_hat_t - x_t
            # PiGDM Laplace Approx (E @ EH is CN x CN)
            sigma_t = sigma_y + sigma_t**2/(1+sigma_t**2)*E@E_H
            # Assume sigma_t is CN x CN, v_t is CN x 1
            v_t = torch.linalg.inv(sigma_t)@(E@x_hat_t.view(N, 1)-y)
            # grad_likelihood is N x 1
            grad_likelihood = -J_t.adjoint()@ E_H@y
            # Reshape grad_likelihood to take image form
            grad_likelihood = grad_likelihood.view(grad_prior.shape)
            # Update step size
            h_t = self.h_0 * (t/1+h_0*(t-1))
            # Update noise injection
            gamma_t = sigma_t*torch.sqrt((1-self.beta*h_t)**2-(1-h_t)**2)
            noise = torch.randn_like(x_t)
            # Stochastic gradient ascent ( sqrt(N) x sqrt(N))
            x_t = x_t + h_t * (grad_prior+grad_likelihood) + gamma_t*noise
            t = t + 1
        return x_t
