import torch 
import torch.fft as fft
import data
import os
import h5py

from utils import saveimg
from mri_utils import mri_encoding, mri_decoding, check_adjoint, detect_acc_mask, make_acc_mask, walsh_smaps
from solvers import conj_grad
from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str, help="Run model over specified test set (provided path to image dir).", default=None)
parser.add_argument("--kspace_path", type =str, help="Corresponding path where kspace data can be found", default = None)

# This will implement SENSE, which essentially performs conjugate gradient on the normal equations for MRI

def eHe(x, mri_encoding, mri_decoding, lam = torch.tensor(0.0000 + 0.000j)):
    # Performs E^H E with lambda regularization
    return mri_decoding(mri_encoding(x)) + lam * x

def sense(y, acceleration_map, smaps, verbose):
    # Build a forward operator out of acceleration_map and smaps
    E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
    EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)
    
    EHE = partial(eHe, mri_encoding = E, mri_decoding = EH)
    # If we have y = Ex, then we want to work with E^Hy = E^HEx, i.e. our symmetric operator is EHE
    EHy = EH(y)

    # Look at kspace to make sure E is correct
    y_fake = E(EHy)
    # Load first slice of y
    saveimg(torch.log10(y_fake[0, :, :].abs()), "kspaceslice1.png")
    saveimg(acceleration_map, "mask.png")
    breakpoint()
    return conj_grad(EHE, EH(y), tol = 1e-6, max_iter = 50, verbose = verbose)

def main(args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    # Take an input argument that specifies a test directory 
    # loader = data.get_data_loader([args.test], load_color=False, test=True, get_smaps = True)
    # Get some sample image 
    fname = 'file_brain_AXT2_210_2100095.h5'
    slice = 3
    #base_fname = os.path.join(args.test, fname)
    #with h5py.File(base_fname) as f:
    #    image = f['image'][slice, :, :]
    #    smaps = f['smaps'][slice, :, :, :]
    # image, smaps, slice, path = next(iter(loader))
    # fname = os.path.basename(path[0])
    # Find the file at the kspace path
    kspace_fname = os.path.join(args.kspace_path, fname)
    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :,  :]
    # Squeeze smaps, also conjugate since they come as conjugated form
    # smaps = smaps[0, :, :, :].conj()
    kspace = torch.from_numpy(kspace)  
    smaps = walsh_smaps(torch.fft.fftshift(torch.fft.ifft2(kspace[None, :, :, :]), dim = (-2, -1)))
    smaps = torch.squeeze(smaps.conj())
    # Detect acceleration maps
    #mask = detect_acc_mask(kspace)
    # Make an acceleration map
    _, mask = make_acc_mask(shape = (smaps.shape[1], smaps.shape[2]), accel = 4, acs_lines = 24)
    # Switch axes and send to GPU
    smaps = smaps.to(device)
    # Normalize smaps for SENSE
    power = torch.sum(torch.abs(smaps)**2, dim=0, keepdim=True)
    # smaps = smaps / torch.sqrt(power + 1e-8)
    kspace = kspace.to(device) 
    mask = mask.to(device)
    # Mask kspace
    kspace_masked = torch.complex(mask[None, :, :], torch.zeros_like(mask[None, :, :])) * kspace
    
    gnd_truth = (mri_decoding(kspace, mask, smaps))
    saveimg(gnd_truth, "EHy.png")
    # Extract a slice of kspace and save it
    kspace_from_gnd_truth = mri_encoding(gnd_truth, torch.ones(mask.shape[0], mask.shape[1], device = device), smaps)
    breakpoint()
    mri_recon, tol_reached = sense(kspace_masked, mask, smaps, verbose = True)

    zero_filled_recon = mri_decoding(kspace_masked, mask, smaps)
    mri_recon = mri_recon
    breakpoint()
    saveimg(zero_filled_recon, "test_zerofilled.png")
    saveimg(mri_recon, "test_sense.png")
    # saveimg(image, "gnd_truth.png")

if __name__ == "__main__":
    """ 
    Load arguments from json file and command line and pass to main.
    """
    args = parser.parse_args()
    main(args)
