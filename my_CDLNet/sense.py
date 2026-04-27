import torch 
import torch.fft as fft
import numpy as np
import data
import os
import h5py

from utils import saveimg
from mri_utils import batched_mri_encoding, batched_mri_decoding, check_adjoint, detect_acc_mask, make_acc_mask, walsh_smaps, fftc, ifftc, mri_awgn
from solvers import conj_grad
from functools import partial
from train import load_model

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kspace_path", type = str, help="Corresponding path where kspace data can be found", default = '../../datasets/fastmri/knee/multicoil_val/file1000031.h5')
parser.add_argument("--noise_level", type = float, help="Std deviation of injected noise into kspace data", default = 0.01)
parser.add_argument("--slice", type = int, help="Slice to kspace to focus on", default = None)

# This will implement SENSE, which essentially performs conjugate gradient on the normal equations for MRI

def eHe(x, mri_encoding, mri_decoding, lam = torch.tensor(0.001 + 0.j)):
    # Performs E^H E with lambda regularization
    return mri_decoding(mri_encoding(x)) + lam * x

def sense(y, acceleration_map, smaps, verbose):
    # Build a forward operator out of acceleration_map and smaps
    E = partial(batched_mri_encoding, mask = acceleration_map, smaps = smaps)
    EH = partial(batched_mri_decoding, mask = acceleration_map, smaps = smaps)
    
    EHE = partial(eHe, mri_encoding = E, mri_decoding = EH)
    # If we have y = Ex, then we want to work with E^Hy = E^HEx, i.e. our symmetric operator is EHE
    EHy = EH(y)
    
    return conj_grad(EHE, EH(y), tol = 1e-6, max_iter = 2000, verbose = verbose)

def main(args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    kspace_fname = args.kspace_path
    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][()]
        print(f.attrs['acquisition'])
    if args.slice is None:
        slice = np.arange(0, kspace.shape[0], 1)
    else:
        slice = [int(args.slice)]
    kspace = torch.from_numpy(kspace)  
    # Load smaps 
    # Search in val dir for corresponding smaps
    fname = os.path.basename(kspace_fname)
    smap_root = "../../datasets/fastmri_preprocessed/knee_coil_combined/pd/val"
    smaps_fname = os.path.join(smap_root, fname)

    with h5py.File(smaps_fname) as f:
        smaps = f['smaps'][()]
        gnd_truth = f['image'][()]
    smaps = torch.from_numpy(smaps)
    gnd_truth = torch.from_numpy(gnd_truth)
    mask = make_acc_mask(shape = (smaps.shape[-2], smaps.shape[-1]), accel = 6, acs_lines = 20)
    # Send to GPU
    smaps = smaps.to(device)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*5e3
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = mask * kspace
    # Try adding some noise to kspace
    noise_level = args.noise_level
    # Don't bother adding additional noise
    # Try on simulated kspace vs measurement kspace
    # Insert channel dim into x
    gnd_truth = gnd_truth.unsqueeze(1).to(device)*5e3
    sim_kspace, sig = mri_awgn(gnd_truth, mask, smaps, noise_level)
    
    lpdsnet_e2e = load_model('trained_nets/knee/LPDS_MRI_Recon_R6_nl1nl2Loss_SimKSpace/args.json', device = device)

    for s in slice:
        mri_recon, tol_reached = sense(sim_kspace[s, :, :, :], mask, smaps[s, :, :, :], verbose = True)
        image = gnd_truth[s, :, :]
        # zero_filled_recon = mri_decoding(kspace_masked, mask, smaps)
        # For our purposes there is no need to compute ground truth
        # gnd_truth = (mri_decoding(kspace, torch.ones(smaps.shape[-2], smaps.shape[-1], device = device), smaps))
        # saveimg(zero_filled_recon, "sense_knee/test_zerofilled.png")
        # For convenience, we can also compute rss
        rss = torch.sqrt(torch.sum(ifftc(kspace[s, :, :, :]).abs()**2, dim = 0))
        lpdsnet_recon, _ = lpdsnet_e2e(sim_kspace[s, :, :, :], sig, mask, smaps[s, :, :, :], mri = True)
        saveimg(mri_recon, "sense_knee/test_sense_slice"+str(s)+".png")
        saveimg(image, "sense_knee/gnd_truth_slice"+str(s)+".png")
        saveimg(rss, "sense_knee/rss_slice"+str(s)+".png")
        saveimg(lpdsnet_recon, "sense_knee/lpdsnet_recon_slice"+str(s)+'.png')

if __name__ == "__main__":
    """ 
    Load arguments from json file and command line and pass to main.
    """
    args = parser.parse_args()
    main(args)
