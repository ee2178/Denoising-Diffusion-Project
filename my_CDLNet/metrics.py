import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import train
import json
import os
import h5py

from pprint import pprint
from diff_model import ImMAP
from mri_utils import make_acc_mask


def psnr(x, y):
    mse = torch.mean((x-y).abs()**2)
    return -10*torch.log10(mse)

def nrmse(x, y):
    rmse = torch.sqrt(torch.mean((x-y).abs()**2))
    dyn_range = torch.max(x.abs()) - torch.min(x.abs())
    return rmse/dyn_range

# Gaussian window
def gaussian_window(size=11, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = torch.outer(g, g)
    # Gaussian window normalization
    w = w / w.sum()
    # reshape to (1,1,k,k) so conv2d works properly
    w = w.view(1, 1, size, size)
    return w

def complex_conv2d(img, weight, window_size):
    # img: N,C,H,W complex
    real = F.conv2d(img.real, weight, padding=window_size//2, groups=img.shape[1])
    imag = F.conv2d(img.imag, weight, padding=window_size//2, groups=img.shape[1])
    
    return torch.complex(real, imag)

def init_e2e(e2e_path, device):
    # Load LPDSNet
    lpds_args_file = open(e2e_path)
    lpds_args = json.load(lpds_args_file)
    lpds_args_file.close()
    lpdsnet, _, _, _ = train.init_model(lpds_args, device = device)
    return lpdsnet

def ssim(x, y, window_size=11):
    """
    Compute SSIM for complex-valued images x and y.
    Shapes: (N, C, H, W), dtype: complex64/complex128
    """
    C1 = 1e-4
    C2 = 9e-4
      
    # 1. Convert complex → magnitude
    x_mag = torch.abs(x)
    y_mag = torch.abs(y)

    # 2. Prepare Gaussian window
    w = gaussian_window(window_size).to(x.device)
    w = w.expand(x.shape[1], 1, window_size, window_size)
    pad = window_size // 2

    # 3. Compute means
    mu_x = F.conv2d(x_mag, w, padding=pad, groups=x.shape[1])
    mu_y = F.conv2d(y_mag, w, padding=pad, groups=x.shape[1])
    mu_x2 = mu_x ** 2
    mu_y2 = mu_y ** 2
    mu_xy = mu_x * mu_y

    # 4. Compute variances
    sigma_x2 = F.conv2d(x_mag * x_mag, w, padding=pad, groups=x.shape[1]) - mu_x2
    sigma_y2 = F.conv2d(y_mag * y_mag, w, padding=pad, groups=x.shape[1]) - mu_y2
    sigma_xy = F.conv2d(x_mag * y_mag, w, padding=pad, groups=x.shape[1]) - mu_xy

    # 5. SSIM map (real-valued)
    num = (2 * mu_xy + C1) * (2 * sigma_xy + C2)
    den = (mu_x2 + mu_y2 + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / (den + 1e-12)

    # 6. Average over spatial dimensions and channels → (N,)
    return ssim_map.mean(dim=(1, 2, 3))


