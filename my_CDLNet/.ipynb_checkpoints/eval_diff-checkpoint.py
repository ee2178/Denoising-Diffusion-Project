import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import train
import time
import json
import os
import h5py

from pprint import pprint
from immap import ImMAP
from diff_testbench import load_model
from mri_utils import make_acc_mask, espirit, ifftc
from metrics import psnr, nrmse, ssim, lpips_alex, joint_normalize

import argparse
import lpips

parser = argparse.ArgumentParser()

parser.add_argument("args_fn", type=str, help="Path to args.json file.", default="config.json")
parser.add_argument("--kspace_path", type = str, help="Corresponding path where kspace data can be found", default = None)
parser.add_argument("--smap_path", type = str, help = "Corresponding path where smap data can be found", default = None)
parser.add_argument("--noise_level", type = float, help="Std deviation of injected noise into kspace data", default = 0.)
parser.add_argument("--slice_range", nargs = 2, type = int, help="Slice range for evaluation per volume", default = [0, 8])
parser.add_argument("--save_name", type = str, help="Name of file to save results to", default="results.txt")
parser.add_argument("--eval_e2e", action='store_true' , help="True if want to evaluate e2e LPDSNet")
parser.add_argument("--immap2p5_path", type = str, help="Corresponding path where immap2p5 args can be found", default = "configs/immap2p5_config.json")
parser.add_argument("--e2enet_path", type = str, help="Corresponding path where mri recon net args can be found", default = "configs/mri_config.json")
parser.add_argument("--immap_mode", type = str, help="Choose the mode of immap we want to evaluate (1, 2, 2.5, 3)", default = "1")
parser.add_argument("--accel", type = int, help="Acceleration factor", default = 6)

args = parser.parse_args()

# We define our own data prep function, slightly different to the other
def prep_data(  kspace_fname,       # Path to kspace
                slice,              # Slice to extract
                acs = 24,           # Default acs size
                accel = 6,          # Acceleration rate
                scale_fac = 2e3,    # Scale factor
                device = 'cpu'      # Device
                ):
    fname = os.path.basename(kspace_fname)

    # Search in val dir for corresponding smaps
    '''
    smaps_fname = os.path.join(smap_root, fname)

    with h5py.File(smaps_fname) as f:
        smaps = f['smaps'][:, :, :, :]
        smaps = smaps[slice, :, :, :]
        # gnd_truth = f['image'][slice, :, :]
    '''
    # At test time, we no longer want to try and grab walsh smaps, we will use espirit on the fly
    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :, :]
    kspace = torch.from_numpy(kspace)

    mask = make_acc_mask(shape = (kspace.shape[-2], kspace.shape[-1]), accel = accel, acs_lines = acs)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*scale_fac
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = mask * kspace

    espirit_smaps = torch.flip(espirit(mask*kspace[None]/scale_fac, acs_size=(acs, acs)), dims = (-2, -1))[0].to(device)
    gnd_truth = (espirit_smaps.conj() * ifftc(kspace)).sum(dim=0).to(device)

    brain_mask = torch.norm(espirit_smaps, dim = 0) != 0

    return kspace, kspace_masked, espirit_smaps, mask, gnd_truth, brain_mask

def compute_nnz_crop(brain_mask, acs_size=(24, 24)):
    nnzs = torch.nonzero(brain_mask*1)
    max_x = torch.max(nnzs[:, 0])
    min_x = torch.min(nnzs[:, 0])

    max_y = torch.max(nnzs[:, 1])
    min_y = torch.min(nnzs[:, 1])
    return min_x, max_x, min_y, max_y

def compute_metrics(args, device):
    noise_level = args.noise_level
    # kspace_fname = args.kspace_path
    # fname = os.path.basename(kspace_fname)
    NRMSE = 0
    PSNR = 0
    SSIM = 0
    LPIPS = 0
    latency = 0
    count = 0
    n_diverged = 0
   
    # Load networks
    # Load networks
    net = load_model(args.args_fn, device = device)
    e2enet = load_model(args.immap2p5_path, device = device)
    if args.e2enet_path is not None:
        lpdsnet_e2e = load_model(args.e2enet_path, device = device)

    immap = ImMAP(net, lam = 10.)

    loss_fn_alex = lpips.LPIPS(net='alex').to(device).eval()

    min_slice = args.slice_range[0]
    max_slice = args.slice_range[1]
    for fname in os.listdir(args.kspace_path):
        with torch.no_grad():
            if fname.startswith('file_brain_AXT2'):
                for slice in range(min_slice, max_slice):
                    kspace, kspace_masked, smaps, mask, gnd_truth, brain_mask = prep_data(os.path.join(args.kspace_path, fname), slice, device = device, accel = args.accel)
                    if not args.eval_e2e:
                        # Use first line for regular immap
                        # recon = immap.forward_2_e2e_conditioned(kspace_masked, noise_level, mask, smaps)
                        if args.immap_mode == '1':
                            current_time = time.time()
                            recon = immap.forward(kspace_masked, noise_level, mask, smaps, verbose = False)
                            final_time = time.time()
                        elif args.immap_mode == '2':
                            current_time = time.time()
                            recon = immap.forward_2(kspace_masked, 0.01, mask, smaps, verbose = False)
                            final_time = time.time()
                        elif args.immap_mode == '2.5':
                            current_time = time.time()
                            recon, _, _ = immap.forward_2p5(kspace_masked, noise_level, mask, smaps, e2enet, verbose = False)
                            final_time = time.time()
                        elif args.immap_mode=='3':
                            current_time = time.time()
                            recon = immap.forward_3(kspace_masked, noise_level, mask, smaps, verbose = False)
                            final_time = time.time()
                        elif args.immap_mode=='3.5':
                            current_time = time.time()
                            recon, _, _ = immap.forward_3p5(kspace_masked, noise_level, mask, smaps, e2enet, verbose = False)
                            final_time = time.time()
                        elif args.immap_mode=='4':
                            # We can have 2 modes here, one using zero filled and the other using a e2enet recon
                            current_time = time.time()
                            if args.e2enet_path is not None:
                                 e2e_recon, _ = lpdsnet_e2e(kspace_masked, noise_level*255., mask = mask[None], smaps = smaps[None], mri = True)
                            else:
                                e2e_recon = None
                            recon = immap.forward_4(kspace_masked, noise_level, mask, smaps, e2enet, recon = e2e_recon, verbose = False)
                            final_time = time.time()
                    else:
                        # This probably doesn't work anymore lol
                        current_time = time.time()
                        recon, _ = lpdsnet_e2e(kspace_masked[None], noise_level*255., mask = mask[None], smaps = smaps[None], mri = True)
                        final_time = time.time()
                    if torch.sum(torch.isnan(recon)) > 0:
                        print(f"{fname} diverged. Skipping this sample")
                        n_diverged = n_diverged + 1
                        break
                    # Apply our brain mask
                    recon = recon * brain_mask
                    
                    # Perform joint normalization prior to computing metrics
                    xp, yp = joint_normalize(recon.abs(), gnd_truth.abs())
                    # Compute our brain mask
                    min_x, max_x, min_y, max_y = compute_nnz_crop(brain_mask)                
                    # Compute PSNR
                    PSNR = PSNR + psnr(xp[brain_mask], yp[brain_mask])
                    # Compute NRMSE
                    NRMSE = NRMSE + nrmse(xp[brain_mask], yp[brain_mask])
                    # Compute SSIM
                    SSIM = SSIM + ssim(xp[None, None, min_x:max_x+1, min_y:max_y+1], yp[None, None, min_x:max_x+1, min_y:max_y+1])
                    # Compute LPIPS
                    LPIPS = LPIPS + lpips_alex(xp[None, None, min_x:max_x+1, min_y:max_y+1], yp[None, None, min_x:max_x+1, min_y:max_y+1], loss_fn_alex)
                    # Compute latency
                    latency = latency + (final_time - current_time)
                    # Increment count
                    count = count + 1

        if count >= (max_slice - min_slice)*100:
            break
    NRMSE = NRMSE / (count-n_diverged)
    PSNR = PSNR / (count-n_diverged)
    SSIM = SSIM / (count-n_diverged)
    LPIPS = LPIPS / (count-n_diverged)
    latency = latency / (count-n_diverged)
    return NRMSE, PSNR, SSIM, LPIPS, latency, n_diverged

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
    nrmse, psnr, ssim, lpips_, latency, n_diverged = compute_metrics(args, device)
    with open(args.save_name, 'w') as f:
        f.write(f'NRMSE: {nrmse}\n')
        f.write(f'PSNR: {psnr} \n')
        f.write(f'SSIM: {ssim} \n')
        f.write(f'LPIPS: {lpips_} \n')
        f.write(f'latency: {latency} \n')
        f.write(f'Diverged: {n_diverged}')

if __name__ == "__main__":
    main(args)
