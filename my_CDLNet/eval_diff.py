import torch
import numpy as np
from diff_model import ImMAP


def main(args):

    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    print(f"Using device {device}.")
    
    # kspace_fname = args.kspace_path
    # fname = os.path.basename(kspace_fname)
    
    for fname in os.listdir(args.kspace_path)
        # Search in val dir for corresponding smaps
        smaps_fname = os.path.join("../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val", fname)
        with h5py.File(smaps_fname) as f:
            smaps = f['smaps'][:, :, :, :]
            smaps = smaps[slice, :, :, :]
            gnd_truth = f['image'][slice, :, :]

        with h5py.File(kspace_fname) as f:
            kspace = f['kspace'][slice, :, :, :]
        # Squeeze smaps, also conjugate since they come as conjugated form
        kspace = torch.from_numpy(kspace)
        smaps = torch.from_numpy(smaps)
        smaps = torch.squeeze(smaps)
        # Detect acceleration maps
        _, mask = make_acc_mask(shape = (smaps.shape[1], smaps.shape[2]), accel = 8, acs_lines = 24)
        # Send to GPU
        smaps = smaps.to(device)
        # Scale kspace and send to GPU
        kspace = kspace.to(device)*2e3
        mask = mask.to(device)
        # Mask kspace
        kspace_masked = mask * kspace
