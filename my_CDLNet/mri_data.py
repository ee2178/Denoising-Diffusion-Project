import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
import argparse
from utils import saveimg
from mri_utils import walsh_smaps, fftc, ifftc

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, help="Run preprocessing over specified training set (provided path to image dir).", default=None)
parser.add_argument("--val", type=str, help="Run preprocessing over specified validation set (provided path to image dir).", default=None)
parser.add_argument("--test", type=str, help="Run preprocessing over specified test set (provided path to image dir).", default=None)
parser.add_argument("--target", type=str, help="Store processed images in a new target directory.", default=None)
ARGS = parser.parse_args()

def crop_center_kspace(kspace, crop_size):
    """
    Crop the central region of k-space.

    Args:
        kspace: complex tensor of shape (B, C, H, W)
        crop_size: int or (h, w)

    Returns:
        Cropped k-space of same shape, with zeros outside the central region.
    """
    B, C, H, W = kspace.shape
    if isinstance(crop_size, int):
        crop_h = crop_w = crop_size
    else:
        crop_h, crop_w = crop_size

    out = torch.zeros_like(kspace)
    ch_start = H // 2 - crop_h // 2
    ch_end = ch_start + crop_h
    cw_start = W // 2 - crop_w // 2
    cw_end = cw_start + crop_w

    out[:, :, ch_start:ch_end, cw_start:cw_end] = kspace[:, :, ch_start:ch_end, cw_start:cw_end]
    return out

def save_volume(kspace, image, smaps, dir, name, target_dir):
    # Save data as hdf5 format as a whole volume
    # Construct the dataset
	# Need to reset indentations
    if dir.endswith('train'):
        split = 'train'
    elif dir.endswith('val'):
        split = 'val'
    elif dir.endswith('test'):
        split = 'test'
    destination = os.path.join(target_dir, split, name)
    with h5py.File(destination, 'w') as f:
        # f.create_dataset('kspace', data=kspace.cpu().numpy())
        f.create_dataset('image', data=image.cpu().numpy())
        f.create_dataset('smaps', data=smaps.cpu().numpy())
    return None

def main(dirs, target_dir):
	# Get device
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")
    for dir in dirs:
        if dir: 
            for name in os.listdir(dir):
				# Only get T2 weighted brain 
                if name.startswith('file_brain_AXT2'):
					# Every file in these directories should be h5 files anyway
                    with h5py.File(os.path.join(dir, name)) as hf:
                        volume_kspace = hf['kspace'][()]
                        # Convert to pytorch tensor (complex valued)
                        volume_kspace = torch.from_numpy(volume_kspace)
                        # Put on GPU
                        volume_kspace = volume_kspace.to(device)
                        # Get kspace centers
                        # volume_kspace_centers = crop_center_kspace(volume_kspace, (640, 24))
                        volume_img = ifftc(volume_kspace)
                        smaps = walsh_smaps(volume_img)
                        # Apply sensitivity maps and then sum
                        volume_combined = torch.einsum('ijkl,ijkl->ikl', smaps.conj(), volume_img)
                        # Save each slice individually
                        save_volume(kspace = volume_kspace, image = volume_combined, smaps = smaps, dir = dir, name = name, target_dir = target_dir)
    return None



if __name__ == "__main__":
    # Iterate through the directories specified
    # Grab a sample k-space volume shaped (n_slices, n_coils, height, width)
    dirs = [ARGS.train, ARGS.val, ARGS.test]
    target_dir = ARGS.target
    print(dirs)
    print(target_dir)
    main(dirs, target_dir)
        
    
