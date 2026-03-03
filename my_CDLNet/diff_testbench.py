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

from mri_utils import mri_encoding, mri_decoding, walsh_smaps, fftc, ifftc, make_acc_mask, quant_complex, quant_tensor, espirit, mri_awgn
from functorch import jacrev, jacfwd
from solvers import conj_grad
from pprint import pprint
from functools import partial
from utils import saveimg
from model_utils import uball_project
from metrics import psnr, ssim
from immap import ImMAP

def eval_immap( immap,      # ImMAP class
                kspace_masked, # noisy masked kspace input
                smaps,      # espirit smaps
                noise_level,# initial noise level 
                mask,       # acceleration mask (uniform)
                brain_mask, # brain mask for more faithful psnr results
                mode,       # ImMAP mode
                gnd_truth,  # Ground truth 
                lpdsnet,    # The network for ImMAP2.5, lpdsnet should be renamed
                init_recon, # Used for warm starts 
                save = False, # Whether or not to save output   
                ):
    """
    mode:
        1   -> forward
        2   -> forward_2
        2.5 -> forward_2p5
        3.5 -> forward_3p5
        4   -> forward_4
    """
    if mode == 1:
        out = immap.forward(
            kspace_masked, noise_level, mask, smaps,
            None, verbose=True
        )
    elif mode == 2:
        out = immap.forward_2(
            kspace_masked[0], noise_level, mask, smaps,
            None, verbose=True
        )
    elif mode == 2.5:
        out, _, _ = immap.forward_2p5(
            kspace_masked[0], noise_level, mask, smaps,
            lpdsnet, save_dir=None, verbose=True, mode=1
        )
    elif mode == 3.5:
        out = immap.forward_3p5(
            kspace_masked, noise_level, mask, smaps,
            lpdsnet, save_dir=None, verbose=True
        )
    elif mode == 4:
        out = immap.forward_4(
            kspace_masked[0], noise_level, mask,
            smaps,
            lpdsnet,
            recon=init_recon,
            save_dir=None,
            verbose=True
        )
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # For ssim, try just zeroing out all the nonmasked pixels?
    # Grab furthest ends of each mask and turn into a square i guess
    nnzs = torch.nonzero(brain_mask*1)
    # Grab max and min across each dimension
    max_x = torch.max(nnzs[:, 0])
    min_x = torch.min(nnzs[:, 0])

    max_y = torch.max(nnzs[:, 1])
    min_y = torch.min(nnzs[:, 1])

    psnr_ = psnr(gnd_truth[brain_mask], out[0, 0, brain_mask])
    ssim_ = ssim(gnd_truth[None, None, min_x:max_x, min_y:max_y], out[:, :, min_x:max_x, min_y:max_y])
    print(f"ImMAP{mode} PSNR:{psnr_}")
    print(f"ImMAP{mode} SSIM:{ssim_}")

    if save == True:
        saveimg(out, "immap"+str(mode)+"out.png", contrast=True)
    return out

def load_model(config, verbose = False, device = 'cpu'):
    # Load Denoiser
    model_args_file = open(config)
    model_args = json.load(model_args_file)
    model_args_file.close()
    if verbose == True:
        pprint(model_args)
    net, _, _, _ = train.init_model(model_args, device=device)
    
    return net

def prep_data(  kspace_fname,       # Path to kspace
                slice,              # Slice to extract
                smap_root = "../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val", # Path to smaps
                noise_level = 0.0,  # Additive noise level
                acs = 24,           # Default acs size
                accel = 6,          # Acceleration rate
                scale_fac = 2e3,    # Scale factor 
                device = 'cpu'      # Device
                ):
    fname = os.path.basename(kspace_fname)

    # Search in val dir for corresponding smaps
    smaps_fname = os.path.join(smap_root, fname)

    with h5py.File(smaps_fname) as f:
        smaps = f['smaps'][:, :, :, :]
        smaps = smaps[slice, :, :, :]
        # gnd_truth = f['image'][slice, :, :]

    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :, :]
        volume_kspace = f['kspace'][()]
    kspace = torch.from_numpy(kspace)
    smaps = torch.from_numpy(smaps)
    smaps = torch.squeeze(smaps)

    mask = make_acc_mask(shape = (smaps.shape[1], smaps.shape[2]), accel = accel, acs_lines = acs)

    # Send to GPU
    smaps = smaps.to(device)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*scale_fac
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = mask * kspace
    
    # We need to use espirit maps to do coil combination
    volume_kspace = torch.from_numpy(volume_kspace)
    volume_kspace = volume_kspace.to(device)

    espirit_smaps = torch.flip(espirit(mask*kspace[None], acs_size=(24, 24)), dims = (-2, -1))[0]
    gnd_truth = (espirit_smaps.conj() * ifftc(kspace)).sum(dim=0)
    
    brain_mask = torch.norm(espirit_smaps, dim = 0) != 0

    return kspace, volume_kspace, smaps, espirit_smaps, mask, gnd_truth, brain_mask

def main():
    # test on one specific sample
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")

    kspace_fname = "../../datasets/fastmri/brain/multicoil_val/file_brain_AXT2_200_2000572.h5"
    # kspace_fname = "../../datasets/fastmri/brain/multicoil_val/file_brain_AXT2_205_2050160.h5"
    
    kspace, volume_kspace, smaps, espirit_smaps, mask, gnd_truth, brain_mask = prep_data(kspace_fname, slice = 5, accel = 6, device = device)
    saveimg(gnd_truth, "gndtruth.png", contrast=True)
    # Load networks
    net = load_model('eval_config.json', device = device)
    net_immap2p5 = load_model('immap2p5_config.json', device = device)
    lpdsnet_e2e = load_model('mri_config.json', device = device)

    # Perform an e2e recon via LPDSNet for warm start
    # noisy_kspace = kspace_masked + noise_level*torch.randn_like(kspace_masked)
    # Add noise in multicoil image space
    noise_level = 0.05
    noisy_kspace, _ = mri_awgn(kspace, mask, espirit_smaps, noise_level, kspace=True)
    e2e_recon, _ = lpdsnet_e2e(noisy_kspace, noise_level*255., mask = mask[None], smaps = espirit_smaps[None], mri = True)

    # Init ImMAP class
    immap = ImMAP(net)
   
    # Generate brain mask 
    modes = [2, 2.5, 4]
    immap_outs = []
    for mode in modes:
        immap_outs.append(eval_immap(immap, noisy_kspace, smaps, noise_level, mask, brain_mask, mode, gnd_truth, net_immap2p5, e2e_recon, save=True)) 

    breakpoint()

if __name__ == "__main__":
    main()
