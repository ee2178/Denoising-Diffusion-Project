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
                save_dir = None
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
        # Try not doing whitening
        '''   
        out = immap.forward_2(
            mask*kspace_white['data'], kspace_white['sigma'].max(), mask, kspace_white['smaps'],
            save_dir = "diff_figs",verbose=True
        )
        out = kspace_white['zinv']*out
        '''
        out = immap.forward_2(
            kspace_masked, 0.01, mask, smaps,
            save_dir = save_dir, verbose=True
        )
        
    elif mode == '2.5':
        _, out, _ = immap.forward_2p5(
            kspace_masked, 0.01, mask, smaps,
            lpdsnet, brain_mask, save_dir=save_dir, verbose=True, mode=1
        )
    elif mode == '3.5':
        out = immap.forward_3p5(
            kspace_masked, noise_level, mask, smaps,
            lpdsnet, save_dir=save_dir, verbose=True
        )
    elif mode == '2.5-WS':
        out = immap.forward_4(
            kspace_masked, noise_level, mask,
            smaps,
            lpdsnet,
            brain_mask,
            recon=init_recon,
            save_dir=save_dir,
            verbose=True,
            sigma_T=0.2
        )
    elif mode == '2-WS':
        # Try nonwhitened kspace first
        out = immap.forward_2(
            kspace_masked, noise_level, mask, smaps,
            save_dir = save_dir, verbose=True, recon = init_recon
        )
        '''
        out = immap.forward_2(
            mask*kspace_white['data'][0], kspace_white['sigma'].max(), mask, kspace_white['smaps'][0],
            None, verbose=True, recon = init_recon
        )
        '''
    elif mode == 'DDS':       
        '''
        out = dds.forward(
            mask*kspace_white['data'], kspace_white['sigma'].max(), mask, kspace_white['smaps'],
            None, verbose=True, sched = 'eero'
        )
        out = kspace_white['zinv']*out
        '''
        out = dds.forward(
            kspace_masked, noise_level, mask, smaps,
            None, verbose=True
        )
        
    else:
        raise ValueError(f"Unknown mode: {mode}")
    # out comes as 4D.

    # For ssim, try just zeroing out all the nonmasked pixels?
    # Grab furthest ends of each mask and turn into a square i guess
    nnzs = torch.nonzero(brain_mask*1)
    # Grab max and min across each dimension
    max_x = torch.max(nnzs[:, 0])
    min_x = torch.min(nnzs[:, 0])

    max_y = torch.max(nnzs[:, 1])
    min_y = torch.min(nnzs[:, 1])

    out = out * brain_mask # Apply the brain mask just to suppress background

    xp, yp = joint_normalize(gnd_truth.abs(), torch.squeeze(out).abs())
    psnr_ = psnr(xp[brain_mask], yp[brain_mask])
    ssim_ = ssim(xp[None, None, min_x:max_x, min_y:max_y], yp[None, None, min_x:max_x, min_y:max_y])
    print(f"ImMAP{mode} PSNR:{psnr_}")
    print(f"ImMAP{mode} SSIM:{ssim_}")
    # Compute error maps 
    err = (gnd_truth - out).abs()*5

    if save == True:
        saveimg(out, "immap"+str(mode)+"out.png", contrast=True)
        saveimg(err, "immap"+str(mode)+"err.png", contrast=True)
    return out

def lambda_sweep(immap, lambda_vec, gnd_truth, kspace_masked, smaps, mask, brain_mask, save_dir = None, init_recon = None):
    outs = []
    psnr_vec = []
    ssim_vec = []
    nnzs = torch.nonzero(brain_mask*1)
    # Grab max and min across each dimension
    max_x = torch.max(nnzs[:, 0])
    min_x = torch.min(nnzs[:, 0])
    max_y = torch.max(nnzs[:, 1])
    min_y = torch.min(nnzs[:, 1])
    for lam in lambda_vec:
        immap.lam = lam
        if init_recon is None:
            out = immap.forward_2(
                kspace_masked, 0.01, mask, smaps,
                save_dir = None, verbose=True
            )

        out = out * brain_mask # Apply the brain mask just to suppress background

        xp, yp = joint_normalize(gnd_truth.abs(), torch.squeeze(out).abs())
        psnr_ = psnr(xp[brain_mask], yp[brain_mask])
        ssim_ = ssim(xp[None, None, min_x:max_x, min_y:max_y], yp[None, None, min_x:max_x, min_y:max_y])
        print(f"ImMAP2 Lambda:{lam} PSNR:{psnr_}")
        print(f"ImMAP2 Lambda:{lam} SSIM:{ssim_}")
        outs.append(out)
        # Compute metrics
        psnr_vec.append(psnr_)
        ssim_vec.append(ssim_)
        if save_dir is not None:
            saveimg(out, save_dir+"/immap2_lam"+str(round(lam.item(), 3))+".png", contrast=True)
    return outs, psnr_vec, ssim_vec

def immap2p5_test(gnd_truth, denoiser, kspace, mask, smaps, net, e2e_recon, noise_level=0.1):
    x = e2e_recon + noise_level * torch.randn_like(e2e_recon)
    x = x.reshape(1, 1, x.shape[-2], x.shape[-1])
    x, _ = denoiser(x, noise_level)
    out,_ = net.forward_double_noise(
        mask*kspace,
        0.001,
        mask,
        smaps,
        x_init=x,
        mri=True,
        sigma_t=noise_level
    )
    saveimg(x, "noisyinput.png", contrast=True)
    saveimg(out, "immap2p5nettest.png", contrast=True)
    return None

def prep_data(  kspace_fname,       # Path to kspace
                slice,              # Slice to extract
                smap_root = "../../datasets/fastmri_preprocessed/knee_coil_combined/pd/val", # Path to smaps
                noise_level = 0.0,  # Additive noise level
                acs = 24,           # Default acs size
                accel = 6,          # Acceleration rate
                scale_fac = 2e3,    # Scale factor 
                device = 'cpu',     # Device
                whiten = False,
                ):
    fname = os.path.basename(kspace_fname)

    # Search in val dir for corresponding smaps
    smaps_fname = os.path.join(smap_root, fname)

    with h5py.File(smaps_fname) as f:
        x = f['image'][()]
        smaps = f['smaps'][slice, :, :, :]
        gnd_truth = f['image'][slice, :, :]

    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :, :]
        volume_kspace = f['kspace'][()]
    kspace = torch.from_numpy(kspace)
    smaps = torch.from_numpy(smaps)
    smaps = torch.squeeze(smaps)

    mask = make_acc_mask(shape = (smaps.shape[1], smaps.shape[2]), accel = accel, acs_lines = acs)

    # Send to GPU
    gnd_truth = torch.from_numpy(gnd_truth).to(device)*scale_fac
    smaps = smaps.to(device)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*scale_fac
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = mask * kspace
    
    # We need to use espirit maps to do coil combination
    volume_kspace = torch.from_numpy(volume_kspace)
    volume_kspace = volume_kspace.to(device)
    
    # Our computed smaps are alredy espirit
    # espirit_smaps = torch.flip(espirit(mask*kspace/scale_fac, acs_size=(acs, acs)), dims = (-2, -1))[0]
    # gnd_truth = (espirit_smaps.conj() * ifftc(kspace)).sum(dim=1)
    
    brain_mask = torch.sum(smaps.abs(), dim = 0) != 0
    # Additionally return a whitened kspace 
    # Add batch dim to kspace and smaps
    kspace = kspace[None]
    smaps = smaps[None, :, :, :]
    if whiten is True:
        kspace_white_dict = whiten(kspace, smaps = smaps)
    else:
        kspace_white_dict = None
    return kspace, kspace_white_dict, smaps, mask, gnd_truth, brain_mask

def main():
    # test on one specific sample
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    # Use knee, not brain
    organ = "knee"

    if organ == "knee":
        kspace_fname = '../../datasets/fastmri/knee/multicoil_val/file1000031.h5'
        smap_root = "../../datasets/fastmri_preprocessed/knee_coil_combined/pd/val"
        slice_ = 25
    elif organ == "brain":
        kspace_fname = "../../datasets/fastmri/brain/multicoil_val/file_brain_AXT2_200_2000572.h5"
        # kspace_fname = "../../datasets/fastmri/brain/multicoil_val/file_brain_AXT2_205_2050160.h5"
        slice_ = 7
        smap_root = "../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val"
    # MRI Params
    accel = 6
    kspace, whitened_kspace, smaps, mask, gnd_truth, brain_mask = prep_data(kspace_fname, smap_root = smap_root, scale_fac = 5e3, slice = slice_, accel = accel, device = device)
    save_extra = True

    # Load networks
    # net = load_model('configs/knee/eval_config.json', device = device)
    net = train.load_model('trained_nets/'+organ+'/LPDSNet/args.json', device = device)
    # net_immap2p5 = load_model('configs/knee/immap2p5_R'+str(accel)+'_config.json', device = device)
    # net_immap2p5 = train.load_model('trained_nets/'+organ+'/LPDS_ImMAP2p5_init_DENOISER_R'+str(accel)+'/args.json', device = device)
    net_immap2p5 = train.load_model('LPDS_ImMAP2p5_SSDU_Masking', device = device)
    # lpdsnet_e2e = load_model('configs/knee/mri_R'+str(accel)+'_config.json', device = device)
    lpdsnet_e2e = train.load_model('trained_nets/'+organ+'/LPDS_MRI_Recon_R'+str(accel)+'_nl1nl2Loss/args.json', device = device)
    # LPDS_MRI_Recon_R6_nl1nl2Loss
    # Perform an e2e recon via LPDSNet for warm start
    # Add noise in multicoil image space if we want
    noise_level = 0.01
    # mri_awgn returns a masked kspace with noise added in the multicoil image domain
    sim_kspace, _ = mri_awgn(gnd_truth, mask, smaps, 0.001)
    meas_kspace = mask*kspace
    
    e2e_recon, _ = lpdsnet_e2e(meas_kspace, noise_level, mask = mask, smaps = smaps, mri = True)

    # Save the e2erecon for comparison
    if save_extra is True:
        saveimg(gnd_truth, "gndtruth.png", contrast=True)
        saveimg(e2e_recon, "e2erecon.png", contrast=True)
    # Init ImMAP class
    # We may want to try a bunch of different lambda values:
    immap = ImMAP(net, lam = 1, beta = 0.1, sigma_L = 0.01)
    dds = DDS(net)
    # Generate brain mask 
    # modes = ['2', '2.5', '2-WS', '2.5-WS', 'DDS']
    modes = ['2.5', '2.5-WS']
    immap_outs = []
    # Make a logscale of lambdas 0.1 to 10
    # lambda_vec = 10**(torch.linspace(0.2, 1, 10))
    # Lambda sweep
    # lambda_sweep(immap, lambda_vec, gnd_truth, meas_kspace, smaps, mask, brain_mask, save_dir = "immap_sweep")
    # save_dir = "diff_figs_immap2p5"
    save_dir = None
    # Try immap with coil combined image plus noise
    # immap2p5_test(gnd_truth, net, kspace, mask, smaps, net_immap2p5, e2e_recon, noise_level=torch.tensor([0.5], device=gnd_truth.device))
    # breakpoint()
    for mode in modes:
        immap_outs.append(eval_immap(immap, dds, meas_kspace, whitened_kspace, smaps, noise_level, mask, brain_mask, mode, gnd_truth, net_immap2p5, e2e_recon, save=True, save_dir = save_dir)) 
    breakpoint()
if __name__ == "__main__":
    main()
