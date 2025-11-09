import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
import argparse
from utils import saveimg

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, help="Run preprocessing over specified training set (provided path to image dir).", default=None)
parser.add_argument("--val", type=str, help="Run preprocessing over specified validation set (provided path to image dir).", default=None)
parser.add_argument("--test", type=str, help="Run preprocessing over specified test set (provided path to image dir).", default=None)
parser.add_argument("--target", type=str, help="Store processed images in a new target directory.", default=None)
ARGS = parser.parse_args()

def walsh_smaps(y: torch.Tensor, ks: int = 5, stride: int = 2):
    """
    Computes coil sensitivity maps using the Walsh method.

    Args:
        y: complex tensor of shape (B, C, H, W)
        ks: patch size
        stride: patch stride

    Returns:
        smaps: sensitivity maps of shape (B, C, H, W)
    """
    B, C, H, W = y.shape

    # Unfold patches: shape (B*C, ks*ks, Npatches)
    # y_reshaped = y.reshape(B * C, 1, H, W)
    unfolded = F.unfold(y, kernel_size=(ks, ks), stride=stride)  # (B, C*ks*ks, Npatches)
    Npatch = unfolded.shape[-1]

    # Reshape to (ks*ks, C, Npatch*B)
    Yp = unfolded.permute(0, 2, 1)
    Yp = Yp.reshape(B*Npatch, ks*ks, C)
    # Covariance matrix X: (Npatch*B, C, C)
    X = Yp.adjoint() @ Yp
    # Reference coil (max signal energy)
    power = y.abs().pow(2).sum(dim=(2, 3, 0))  # (C,)
    Cref = power.argmax().item()

    Q, _, V = torch.linalg.svd(X, full_matrices=False)
    Q = V[:, 0, :]  # (Npatch*B, C)

    Q = Q.view(B, Npatch, C) # (B, Npatch, C)
    Q = Q.permute(2, 0, 1) # (C, B, Npatch)

    # Align phase using reference coil
    qref = Q[Cref:Cref + 1, :, :]
    Q *= qref.conj().sgn()

    # Reshape to low-res map: (B, C, Hp, Wp)
    Hp = (H - ks) // stride + 1
    Wp = (W - ks) // stride + 1
    smaps_p = Q.permute(1, 0, 2).reshape(B, C, Hp, Wp)

    # Upsample to full size
    smaps_real = F.interpolate(smaps_p.real, size=(H, W), mode='bilinear', align_corners=False)
    smaps_imag = F.interpolate(smaps_p.imag, size=(H, W), mode='bilinear', align_corners=False)
    smaps = torch.complex(smaps_real, smaps_imag)
    # Normalize
    norm = smaps.abs().pow(2).sum(dim=1, keepdim=True)
    smaps /= (norm.sqrt() + 1e-6)

    return smaps.conj()
# Perform zero filled reconstruction on the center of kspace
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
        f.create_dataset('kspace', data=kspace.cpu().numpy())
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
                    hf = h5py.File(os.path.join(dir, name))
                    volume_kspace = hf['kspace'][()]
                    # Convert to pytorch tensor (complex valued)
                    volume_kspace = torch.from_numpy(volume_kspace)
                    # Put on GPU
                    volume_kspace = volume_kspace.to(device)
                    # Get kspace centers
                    volume_kspace_centers = crop_center_kspace(volume_kspace, (640, 24))
                    volume_img_centers = torch.fft.fftshift(torch.fft.ifft2(volume_kspace))
                    smaps = walsh_smaps(volume_img_centers)
                    # Apply sensitivity maps and then sum
                    volume_combined = torch.einsum('ijkl,ijkl->ikl', volume_img_centers, smaps)
                    breakpoint()
                    # Save each slice individually
                    save_volume(volume_kspace, volume_combined, smaps, dir, name, target_dir)
    return None

if __name__ == "__main__":
    # Iterate through the directories specified
    # Grab a sample k-space volume shaped (n_slices, n_coils, height, width)
    dirs = [ARGS.train, ARGS.val, ARGS.test]
    target_dir = ARGS.target
    print(dirs)
    print(target_dir)
    main(dirs, target_dir)
        
    
