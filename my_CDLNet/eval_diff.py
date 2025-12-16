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

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("args_fn", type=str, help="Path to args.json file.", default="config.json")
parser.add_argument("--kspace_path", type = str, help="Corresponding path where kspace data can be found", default = None)
parser.add_argument("--smap_path", type = str, help = "Corresponding path where smap data can be found", default = None)
parser.add_argument("--noise_level", type = float, help="Std deviation of injected noise into kspace data", default = 0.)
parser.add_argument("--slice_range", type = tuple, help="Slice range for evaluation per volume", default = (0, 8))
parser.add_argument("--save_name", type = str, help="Name of file to save results to", default="results.txt")

args = parser.parse_args()

def prep_data(fname, device, slice, smap_path, kspace_path, K = 2e3):
    smaps_fname = os.path.join(smap_path, fname)
    with h5py.File(smaps_fname) as f:
        smaps = f['smaps'][:, :, :, :]
        smaps = smaps[slice, :, :, :]
        gnd_truth = f['image'][slice, :, :]
    kspace_fname = os.path.join(kspace_path, fname)
    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :, :]
    # Squeeze smaps, also conjugate since they come as conjugated form
    kspace = torch.from_numpy(kspace)
    smaps = torch.from_numpy(smaps)
    smaps = torch.squeeze(smaps)
    gnd_truth = torch.from_numpy(gnd_truth)
    # Detect acceleration maps
    _, mask = make_acc_mask(shape = (smaps.shape[1], smaps.shape[2]), accel = 8, acs_lines = 24)
    # Send to GPU
    smaps = smaps.to(device)
    gnd_truth = gnd_truth.to(device)*K
    # Append dimensions to make gnd_truth 4D
    gnd_truth = gnd_truth[None, None]
    kspace = kspace.to(device)*K
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = mask * kspace

    return kspace, kspace_masked, mask, smaps, gnd_truth

def compute_metrics(args, device):
    # Load denoiser
    model_args_file = open(args.args_fn)
    model_args = json.load(model_args_file)
    pprint(model_args)

    noise_level = args.noise_level
    # kspace_fname = args.kspace_path
    # fname = os.path.basename(kspace_fname)
    NRMSE = 0
    PSNR = 0
    SSIM = 0
    count = 0
    n_diverged = 0
    # Load ImMAP 
    net, _, _, _= train.init_model(model_args, device=device, quant_ckpt = True)
    net.eval()
    immap = ImMAP(net)

    min_slice = args.slice_range[0]
    max_slice = args.slice_range[1]
    for fname in os.listdir(args.kspace_path):
        if fname.startswith('file_brain_AXT2'):
            for slice in range(min_slice, max_slice):
                kspace, kspace_masked, mask, smaps, gnd_truth = prep_data(fname, device, slice, args.smap_path, args.kspace_path)
                recon = immap.forward_quant_smaps(kspace_masked, noise_level, mask, smaps, mode = 1)
                if torch.sum(torch.isnan(recon)) > 0:
                    print(f"{fname} diverged. Skipping this sample")
                    n_diverged = n_diverged + 1
                    break
                # Compute PSNR
                PSNR = PSNR + psnr(recon, gnd_truth)
                # Compute NRMSE
                NRMSE = NRMSE + nrmse(gnd_truth, recon)
                # Compute SSIM
                SSIM = SSIM + ssim(recon, gnd_truth)
                # Increment count
                count = count + 1

        if count >= (max_slice - min_slice)*5:
            break
    NRMSE = NRMSE / (count-n_diverged)
    PSNR = PSNR / (count-n_diverged)
    SSIM = SSIM / (count-n_diverged)
    return NRMSE, PSNR, SSIM, n_diverged

def main(args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    nrmse, psnr, ssim, n_diverged = compute_metrics(args, device)
    with open(args.save_name, 'w') as f:
        f.write(f'NRMSE: {nrmse}\n')
        f.write(f'PSNR: {psnr} \n')
        f.write(f'SSIM: {ssim} \n')
        f.write(f'Diverged: {n_diverged}')

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



if __name__ == "__main__":
    main(args)
