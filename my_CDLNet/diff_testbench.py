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
from metrics import joint_normalize, psnr, ssim
from immap import ImMAP
from dds import DDS
from nle import whiten 

def eval_immap( immap,      # ImMAP class
                dds,        # DDS class
                kspace_masked, # noisy masked kspace input
                kspace_white, # dict of whiten(kspace) outputs
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
    if mode == '1':
        out = immap.forward(
            kspace_masked, noise_level, mask, smaps,
            None, verbose=True
        )
    elif mode == '2':
        # In mode 2 let's using whitening kspace first
        out = immap.forward_2(
            mask*kspace_white['data'][0], kspace_white['sigma'].max(), mask, kspace_white['smaps'][0],
            None, verbose=True
        )
        out = kspace_white['zinv']*out
    elif mode == '2.5':
        out, _, _ = immap.forward_2p5(
            kspace_masked, noise_level, mask, smaps,
            lpdsnet, save_dir=None, verbose=True, mode=1
        )
    elif mode == '3.5':
        out = immap.forward_3p5(
            kspace_masked, noise_level, mask, smaps,
            lpdsnet, save_dir=None, verbose=True
        )
    elif mode == '2.5-WS':
        out = immap.forward_4(
            kspace_masked, noise_level, mask,
            smaps,
            lpdsnet,
            recon=init_recon,
            save_dir=None,
            verbose=True
        )
    elif mode == '2-WS':
        # Try nonwhitened kspace first
        out = immap.forward_2(
            kspace_masked, 0.01, mask, smaps,
            None, verbose=True, recon = init_recon
        )
        '''
        out = immap.forward_2(
            mask*kspace_white['data'][0], kspace_white['sigma'].max(), mask, kspace_white['smaps'][0],
            None, verbose=True, recon = init_recon
        )
        '''
    elif mode == 'DDS':       
        out = dds.forward(
            mask*kspace_white['data'][0], kspace_white['sigma'].max(), mask, kspace_white['smaps'][0],
            None, verbose=True, sched = 'eero'
        )
        out = kspace_white['zinv']*out
        '''
        out = dds.forward(
            kspace_masked, noise_level, mask, smaps,
            None, verbose=True
        )
        '''
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

    out = out * brain_mask # Apply the brain mask just to suppress background

    xp, yp = joint_normalize(gnd_truth.abs(), out[0,0].abs())
    psnr_ = psnr(xp[brain_mask], yp[brain_mask])
    ssim_ = ssim(xp[None, None, min_x:max_x, min_y:max_y], yp[None, None, min_x:max_x, min_y:max_y])
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
                smap_root = "../../datasets/fastmri_preprocessed/knee_coil_combined/pd/val", # Path to smaps
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

    espirit_smaps = torch.flip(espirit(mask*kspace[None]/scale_fac, acs_size=(acs, acs)), dims = (-2, -1))[0]
    gnd_truth = (espirit_smaps.conj() * ifftc(kspace)).sum(dim=0)
    
    brain_mask = torch.norm(espirit_smaps, dim = 0) != 0

    # Additionally return a whitened kspace 
    kspace_white_dict = whiten(kspace[None], smaps = espirit_smaps[None])

    return kspace, kspace_white_dict, smaps, espirit_smaps, mask, gnd_truth, brain_mask

def main():
    # test on one specific sample
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    # Use knee, not brain
    kspace_fname = '../../datasets/fastmri/knee/multicoil_val/file1000323.h5'
    # kspace_fname = "../../datasets/fastmri/brain/multicoil_val/file_brain_AXT2_200_2000572.h5"
    # kspace_fname = "../../datasets/fastmri/brain/multicoil_val/file_brain_AXT2_205_2050160.h5"
    # MRI Params
    accel = 6
    slice_ = 17
    kspace, whitened_kspace, smaps, espirit_smaps, mask, gnd_truth, brain_mask = prep_data(kspace_fname, slice = slice_, accel = accel, device = device)
    saveimg(gnd_truth, "gndtruth.png", contrast=True)
    # Load networks
    net = load_model('configs/knee/eval_config.json', device = device)
    net_immap2p5 = load_model('configs/knee/immap2p5_R'+str(accel)+'_config.json', device = device)
    lpdsnet_e2e = load_model('configs/knee/mri_R'+str(accel)+'_config.json', device = device)

    # Perform an e2e recon via LPDSNet for warm start
    # Add noise in multicoil image space if we want
    noise_level = 0.00
    # mri_awgn returns a masked kspace with noise added in the multicoil image domain
    noisy_kspace, _ = mri_awgn(gnd_truth, mask, espirit_smaps, noise_level)
    e2e_recon, _ = lpdsnet_e2e(noisy_kspace, noise_level*255., mask = mask[None], smaps = espirit_smaps[None], mri = True)

    # Save the e2erecon for comparison
    saveimg(e2e_recon, "e2erecon.png", contrast=True)

    # Init ImMAP class
    # We may want to try a bunch of different lambda values:
    immap = ImMAP(net, lam = 10)
    dds = DDS(net)
    # Generate brain mask 
    modes = ['2', '2.5', '2-WS', '2.5-WS', 'DDS']
    # modes = ['2','2-WS']
    immap_outs = []
    for mode in modes:
        immap_outs.append(eval_immap(immap, dds, noisy_kspace, whitened_kspace, espirit_smaps, noise_level, mask, brain_mask, mode, gnd_truth, net_immap2p5, e2e_recon, save=True)) 

if __name__ == "__main__":
    main()
