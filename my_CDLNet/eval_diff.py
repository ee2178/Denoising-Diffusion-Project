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
from mri_utils import make_acc_mask, espirit
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
parser.add_argument("--immap_mode", type = str, help="Choose the mode of immap we want to evaluate (1, 2, 2.5, 3)", default = 1)

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

def compute_brain_mask(masked_kspace, acs_size=(24, 24)):
    espirit_smaps = torch.flip(espirit(masked_kspace), dims = (-2, -1))
    brain_mask = torch.norm(espirit_smaps[0], dim = 0) != 0

    nnzs = torch.nonzero(brain_mask*1)
    max_x = torch.max(nnzs[:, 0])
    min_x = torch.min(nnzs[:, 0])

    max_y = torch.max(nnzs[:, 1])
    min_y = torch.min(nnzs[:, 1])
    return brain_mask, min_x, max_x, min_y, max_y

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
                        if args.immap_mode == '1':
                            recon = immap.forward(kspace_masked, noise_level, mask, smaps)
                        elif args.immap_mode == '2':
                            recon = immap.forward_2(kspace_masked, noise_level, mask, smaps)
                        elif args.immap_mode == '2.5':
                            recon, _, _ = immap.forward_2p5(kspace_masked, noise_level, mask, smaps, e2enet, mode=1)
                        elif args.immap_mode=='3':
                            recon = immap.forward_3(kspace_masked, noise_level, mask, smaps)
                        elif args.immap_mode=='3.5':
                            recon, _, _ = immap.forward_3p5(kspace_masked, noise_level, mask, smaps, e2enet)
                    else:
                        # This probably doesn't work anymore lol
                        recon, _ = e2enet(kspace_masked[None], noise_level*255., mask = mask[None], smaps = smaps[None], mri = True)
                    if torch.sum(torch.isnan(recon)) > 0:
                        print(f"{fname} diverged. Skipping this sample")
                        n_diverged = n_diverged + 1
                        break
                    # Compute our brain mask
                    brain_mask, min_x, max_x, min_y, max_y = compute_brain_mask(kspace_masked[None])                
                    # Compute PSNR
                    PSNR = PSNR + psnr(recon[0,0,brain_mask], gnd_truth[0,0,brain_mask])
                    # Compute NRMSE
                    NRMSE = NRMSE + nrmse(gnd_truth[0,0,brain_mask], recon[0,0,brain_mask])
                    # Compute SSIM
                    SSIM = SSIM + ssim(recon[:, :, min_x:max_x, min_y:max_y], gnd_truth[:, :, min_x:max_x, min_y:max_y])
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
