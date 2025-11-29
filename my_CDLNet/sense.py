import torch 
import torch.fft as fft
import data
import os
import h5py

from utils import saveimg
from mri_utils import mri_encoding, mri_decoding, check_adjoint, detect_acc_mask, make_acc_mask, walsh_smaps, fftc, ifftc
from solvers import conj_grad
from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--kspace_path", type = str, help="Corresponding path where kspace data can be found", default = None)
parser.add_argument("--noise_level", type = float, help="Std deviation of injected noise into kspace data", default = 0.)

# This will implement SENSE, which essentially performs conjugate gradient on the normal equations for MRI

def eHe(x, mri_encoding, mri_decoding, lam = torch.tensor(0.0001 + 0.000j)):
    # Performs E^H E with lambda regularization
    return mri_decoding(mri_encoding(x)) + lam * x

def sense(y, acceleration_map, smaps, verbose):
    # Build a forward operator out of acceleration_map and smaps
    E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
    EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)
    
    EHE = partial(eHe, mri_encoding = E, mri_decoding = EH)
    # If we have y = Ex, then we want to work with E^Hy = E^HEx, i.e. our symmetric operator is EHE
    EHy = EH(y)
    
    return conj_grad(EHE, EH(y), tol = 1e-6, max_iter = 50, verbose = verbose)

def main(args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    
    slice = 3
    kspace_fname = args.kspace_path
    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :, :]
    # Squeeze smaps, also conjugate since they come as conjugated form
    # smaps = smaps[0, :, :, :].conj()
    kspace = torch.from_numpy(kspace)  
    smaps = walsh_smaps(ifftc(kspace[None]))
    smaps = torch.squeeze(smaps.conj())
    # Detect acceleration maps
    #mask = detect_acc_mask(kspace)

    _, mask = make_acc_mask(shape = (smaps.shape[1], smaps.shape[2]), accel = 2, acs_lines = 24)
    # Send to GPU
    smaps = smaps.to(device)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*2e2
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = mask * kspace
    # Try adding some noise to kspace
    noise_level = args.noise_level
    kspace_masked = kspace_masked + noise_level*torch.randn_like(kspace_masked)
    

    gnd_truth = (mri_decoding(kspace, torch.ones(smaps.shape[1], smaps.shape[2], device = device), smaps))
    saveimg(gnd_truth, "EHy.png")
    mri_recon, tol_reached = sense(kspace_masked, mask, smaps, verbose = True)
    zero_filled_recon = mri_decoding(kspace_masked, mask, smaps)

    saveimg(zero_filled_recon, "test_zerofilled.png")
    saveimg(mri_recon, "test_sense.png")
    # saveimg(image, "gnd_truth.png")

if __name__ == "__main__":
    """ 
    Load arguments from json file and command line and pass to main.
    """
    args = parser.parse_args()
    main(args)
