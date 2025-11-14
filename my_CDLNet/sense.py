import torch 
import torch.fft as fft
import data
import os
import h5py

from utils import saveimg
from mri_utils import mri_encoding, mri_decoding, detect_acc_mask, make_acc_mask
from solvers import conj_grad
from functools import partial

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--test", type=str, help="Run model over specified test set (provided path to image dir).", default=None)
parser.add_argument("--kspace_path", type =str, help="Corresponding path where kspace data can be found", default = None)

# This will implement SENSE, which essentially performs conjugate gradient on the normal equations for MRI

def eHe(x, mri_encoding, mri_decoding):
    # Performs E^H E
    return mri_decoding(mri_encoding(x))

def sense(y, acceleration_map, smaps):
    # Build a forward operator out of acceleration_map and smaps
    E = partial(mri_encoding, acceleration_map = acceleration_map, smaps = smaps)
    EH = partial(mri_decoding, acceleration_map = acceleration_map, smaps = smaps)
    
    EHE = partial(eHe, mri_encoding = E, mri_decoding = EH)
    # If we have y = Ex, then we want to work with E^Hy = E^HEx, i.e. our symmetric operator is EHE
    EHy = EH(y)
    return conj_grad(EHE, EH(y), tol = 1e-8, max_iter = 1e4)

def main(args):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    # Take an input argument that specifies a test directory 
    loader = data.get_data_loader([args.test], load_color=False, test=True, get_smaps = True)
    # Get some sample image 
    smaps, slice, path = next(iter(loader))
    fname = os.path.basename(path[0])
    # Find the file at the kspace path
    kspace_fname = os.path.join(args.kspace_path, fname)
    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][slice, :, :,  :]
    # Squeeze smaps
    smaps = smaps[0, :, :, :]
    kspace = torch.from_numpy(kspace)  

    # Detect acceleration maps
    #mask = detect_acc_mask(kspace)
    
    # Make an acceleration map
    mask, _ = make_acc_mask(shape = (smaps.shape[2], smaps.shape[2]), accel = 4, acs_lines = 24)
    # Switch axes and send to GPU
    smaps = smaps.permute(0, 2, 1).to(device)
    # Normalize smaps for SENSE
    smaps = smaps / torch.norm(smaps, dim = (1, 2), keepdim = True)
    kspace = kspace.permute(0, 2, 1).to(device)*1e6
    mask = mask.to(device)

    # Mask kspace
    kspace_masked = torch.einsum('jj, ijk -> ijk', mask, kspace)
    breakpoint()
    mri_recon, tol_reached = sense(kspace_masked, mask, smaps)

    breakpoint()
    gnd_truth = mri_decoding(kspace, mask, smaps)
    zero_filled_recon = mri_decoding(kspace_masked, mask, smaps)
    saveimg(zero_filled_recon, "test_zerofilled.png")
    saveimg(mri_recon, "test_sense.png")
    saveimg(gnd_truth, "gnd_truth.png")
    
if __name__ == "__main__":
    """ 
    Load arguments from json file and command line and pass to main.
    """
    args = parser.parse_args()
    main(args)
