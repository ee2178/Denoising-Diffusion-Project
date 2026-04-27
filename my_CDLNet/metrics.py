import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import train
import json
import os
import h5py
import lpips

from pprint import pprint
from mri_utils import make_acc_mask
from transforms import gaussian_window

def joint_normalize(x, y):
    # Let's perform a normalization of two images based on their joint statistics
    # Assume x and y are same shape and magnitude images 
    x = x.squeeze()
    y = y.squeeze()
    xy = torch.cat((x, y), dim = 0)
    min_xy = xy.min()
    max_xy = xy.max()
    xp = (x-min_xy)/(max_xy-min_xy)
    yp = (y-min_xy)/(max_xy-min_xy)

    return xp, yp

def lpips_alex(x, y, loss_fn):
    # We need to renormalize our images and then i guess turn them into 3 channel things
    # We can cheat a little bit, let's assume inputs are [0, 1] grayscale. Then, let's just do -0.5 * 2
    xp = (x - 0.5)*2
    yp = (y - 0.5)*2

    xp = xp.expand(-1, 3, -1, -1)
    yp = yp.expand(-1, 3, -1, -1)

    #Compute loss
    with torch.no_grad():
        loss = loss_fn(xp, yp)
    return loss


def psnr(x, y):
    mse = torch.mean((x-y).abs()**2)
    return -10*torch.log10(mse)

def nrmse(x, y):
    rmse = torch.sqrt(torch.mean((x-y).abs()**2))
    # Be careful here, we should probably use the max and min across x and y
    xy = torch.cat((x, y), dim = 0)
    min_xy = xy.min()
    max_xy = xy.max()
    dyn_range = torch.max(max_xy - min_xy)
    return rmse/dyn_range

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
    C1 = (1e-2)**2
    C2 = (3e-2)**2
      
    # 1. Convert complex → magnitude
    x_mag = torch.abs(x)
    y_mag = torch.abs(y)

    # 2. Prepare Gaussian window
    width = int((window_size-1)/2)
    w = gaussian_window(width).to(x.device)
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


