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
from metrics import psnr, nrmse, ssim

import argparse

parser = argparse.ArgumentParser()

parser.add_argument("args_fn", type=str, help="Path to args.json file.", default="config.json")
parser.add_argument("--kspace_path", type = str, help="Corresponding path where kspace data can be found", default = None)
parser.add_argument("--smap_path", type = str, help = "Corresponding path where smap data can be found", default = None)
parser.add_argument("--noise_level", type = float, help="Std deviation of injected noise into kspace data", default = 0.)
parser.add_argument("--slice_range", type = tuple, help="Slice range for evaluation per volume", default = (0, 8))
parser.add_argument("--save_name", type = str, help="Name of file to save results to", default="results.txt")
parser.add_argument("--eval_e2e", type = str, help="True if want to evaluate e2e LPDSNet", default=False)
parser.add_argument("--e2e_path", type = str, help="Corresponding path where e2enet args can be found", default = "mri_config.json")

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
    
    # Load e2e LPDSNet
    e2enet = init_e2e(args.e2e_path, device=device)
    min_slice = args.slice_range[0]
    max_slice = args.slice_range[1]
    for fname in os.listdir(args.kspace_path):
        with torch.no_grad():
            if fname.startswith('file_brain_AXT2'):
                for slice in range(min_slice, max_slice):
                    kspace, kspace_masked, mask, smaps, gnd_truth = prep_data(fname, device, slice, args.smap_path, args.kspace_path)
                    if args.eval_e2e == 'False':
                        # Use first line for regular immap
                        # recon = immap.forward_2_e2e_conditioned(kspace_masked, noise_level, mask, smaps)
                        recon = immap.forward_2_e2econditioned(kspace_masked, noise_level, mask, smaps, e2enet, mode=2)
                    else:
                        # This probably doesn't work anymore lol
                        recon, _ = e2enet(kspace_masked[None], noise_level*255., mask = mask[None], smaps = smaps[None], mri = True)
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

def init_e2e(e2e_path, device):
    # Load LPDSNet
    lpds_args_file = open(e2e_path)
    lpds_args = json.load(lpds_args_file)
    lpds_args_file.close()
    lpdsnet, _, _, _ = train.init_model(lpds_args, device = device)
    return lpdsnet

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

if __name__ == "__main__":
    main(args)
