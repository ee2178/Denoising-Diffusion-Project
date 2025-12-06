import torch
import numpy as np
from diff_model import ImMAP

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("args_fn", type=str, help="Path to args.json file.", default="config.json")
parser.add_argument("--kspace_path", type = str, help="Corresponding path where kspace data can be found", default = None)
parser.add_argument("--smap_path", type = str, help = "Corresponding path where smap data can be found", default = None)
parser.add_argument("--noise_level", type = float, help="Std deviation of injected noise into kspace data", default = 0.)
parser.add_argument("--slice_range", type = tuple, help="Slice range for evaluation per volume", default = (0, 8))
parser.add_argument("--save_name", type = str, help="Name of file to save results to", default="results.txt")

def prep_data(fname, device, slice, smap_path, K = 2e3):
    smaps_fname = os.path.join(smap_path, fname)
    with h5py.File(smaps_fname) as f:
        smaps = f['smaps'][:, :, :, :]
        smaps = smaps[slice, :, :, :]
        gnd_truth = f['image'][slice, :, :]

    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :, :]
    # Squeeze smaps, also conjugate since they come as conjugated form
    kspace = torch.from_numpy(kspace)
    smaps = torch.from_numpy(smaps)
    smaps = torch.squeeze(smaps)
    # Detect acceleration maps
    _, mask = make_acc_mask(shape = (smaps.shape[1], smaps.shape[2]), accel = 8, acs_lines = 24)
    # Send to GPU
    smaps = smaps.to(device)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*K
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = mask * kspace
    # Scale gnd_truth, send to tensor
    gnd_truth = torch.from_numpy(gnd_truth)*K
    return kspace, kspace_masked, mask, smaps, gnd_truth

def main(args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    
    # Load model 
    net, _, _, epoch0 = train.init_model(model_args, device=device)
    net.eval()
    immap = ImMAP(net)

    noise_level = args.noise_level
    # kspace_fname = args.kspace_path
    # fname = os.path.basename(kspace_fname)
    PSNR = 0
    SSIM = 0

    min_slice = args.slice_range[0]
    max_slice = args.slice_range[1]
    for fname in os.listdir(args.kspace_path):
        for slice in range(min_slice, max_slice):
            kspace, kspace_masked, mask, smaps, gnd_truth = prep_data(fname)
            recon = immap(kspace_masked, noise_level, mask, smaps)
        
            # Compute PSNR
            PSNR = PSNR + psnr(recon, gnd_truth)
            # Compute SSIM
            SSIM = SSIM + ssim(recon, gnd_truth)
            # Increment count
            count = count + 1

            if count <= (max_slice - min_slice)*10:
                break
    
    PSNR = PSNR / count
    SSIM = SSIM / count 

    with open('results.txt', 'w') as f:
        f.write(f'PSNR: {PSNR} \n')
        f.write(f'SSIM: {SSIM}')

def psnr(x, y):
    mse = torch.mean((x-y).abs()**2)
    return -10*torch.log10(mse)

# Gaussian window
def gaussian_window(size, sigma=1.5):
    coords = torch.arange(size, dtype=torch.float32) - size // 2
    g = torch.exp(-(coords ** 2) / (2 * sigma * sigma))
    g = g / g.sum()
    w = torch.outer(g, g)
    return w

def complex_conv2d(img, weight):
    # img: N,C,H,W complex
    real = F.conv2d(img.real, weight, padding=window_size//2, groups=img.shape[1])
    imag = F.conv2d(img.imag, weight, padding=window_size//2, groups=img.shape[1])
    return torch.complex(real, imag)

def ssim(x, y):
    """
    Compute SSI
    M for complex-valued images x and y.
    Shapes: (N, C, H, W), dtype: complex64/complex128
    """
    # Convolve real & imaginary parts together using complex arithmetic
    w = gaussian_window(window_size).to(x.device)
    w = w.expand(x.shape[1], 1, window_size, window_size)
    #  Means
    mu_x = complex_conv2d(x, w)
    mu_y = complex_conv2d(y, w)

    # Variances and covariance (complex version)
    sigma_x2 = complex_conv2d(x * x.conj(), w) - mu_x * mu_x.conj()
    sigma_y2 = complex_conv2d(y * y.conj(), w) - mu_y * mu_y.conj()
    sigma_xy = complex_conv2d(x * y.conj(), w) - mu_x * mu_y.conj()

    # SSIM for complex images (complex numerator & denominator)
    num = (2 * mu_x * mu_y.conj() + C1) * (2 * sigma_xy + C2)
    den = (mu_x * mu_x.conj() + mu_y * mu_y.conj() + C1) * (sigma_x2 + sigma_y2 + C2)
    ssim_map = num / den

    # Return mean magnitude (real-valued similarity)
    return ssim_map.abs().mean()
