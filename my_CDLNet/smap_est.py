import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
import math
import argparse
from utils import saveimg
from mri_utils import walsh_smaps, fftc, ifftc, espirit

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, help="Run preprocessing over specified training set (provided path to image dir).", default=None)
parser.add_argument("--val", type=str, help="Run preprocessing over specified validation set (provided path to image dir).", default=None)
parser.add_argument("--test", type=str, help="Run preprocessing over specified test set (provided path to image dir).", default=None)
parser.add_argument("--target", type=str, help="Store processed images in a new target directory.", default=None)
parser.add_argument("--method", type=str, help="Choose smap estimation algorithm", default="espirit")
parser.add_argument("--overwrite", action="store_true")

ARGS = parser.parse_args()

def crop_center_kspace(kspace, crop_size):
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
    if dir.endswith('train'):
        split = 'train'
    elif dir.endswith('val'):
        split = 'val'
    elif dir.endswith('test'):
        split = 'test'
    elif dir.endswith('test_v2'):
        split = 'test'
    destination = os.path.join(target_dir, split, name)
    with h5py.File(destination, 'w') as f:
        # f.create_dataset('kspace', data=kspace.cpu().numpy())
        # f.create_dataset('image', data=image.cpu().numpy())
        f.create_dataset('smaps', data=smaps.cpu().numpy())
    return None

def main(dirs, target_dir, method, batch_size=2):
    ngpu = torch.cuda.device_count()
    device = torch.device("cuda:0" if ngpu > 0 else "cpu")

    for dir in dirs:
        if not dir:
            continue

        split = os.path.basename(os.path.normpath(dir))  # e.g. train / val / test

        for name in os.listdir(dir):
            if not name.startswith("file"):
                continue

            # ---- Construct destination path (must match save_volume logic)
            out_dir = os.path.join(target_dir, split)
            out_path = os.path.join(out_dir, name)

            # ---- Skip if already processed
            if os.path.exists(out_path) and not ARGS.overwrite:
                print(f"Skipping {name} (already exists)")
                continue

            in_path = os.path.join(dir, name)

            with h5py.File(in_path, "r") as hf:
                volume_kspace = torch.from_numpy(hf["kspace"][()]).to(device, non_blocking=True)
                assert volume_kspace.is_cuda

                with torch.inference_mode():
                    if method == "walsh":
                        volume_img = ifftc(volume_kspace)
                        smaps = walsh_smaps(volume_img)
                        smaps_cpu = smaps.detach().cpu()
                        del volume_img, smaps

                    elif method == "espirit":
                        S = volume_kspace.shape[0]
                        smaps_cpu_chunks = []

                        for s0 in range(0, S, batch_size):
                            s1 = min(S, s0 + batch_size)

                            ks_batch = volume_kspace[s0:s1]

                            sm_batch = espirit(
                                ks_batch,
                                acs_size=(24, 24),
                                thresh_rowspace=0.01,
                                thresh_eig=0.99,
                            )
                            sm_batch = torch.flip(sm_batch, dims=(-2, -1))

                            smaps_cpu_chunks.append(sm_batch.detach().cpu())

                            del ks_batch, sm_batch

                        smaps_cpu = torch.cat(smaps_cpu_chunks, dim=0)
                        del smaps_cpu_chunks

                    else:
                        raise ValueError(f"Unknown method: {method}")

                save_volume(
                    kspace=volume_kspace.detach().cpu(),
                    image=None,
                    smaps=smaps_cpu,
                    dir=dir,
                    name=name,
                    target_dir=target_dir,
                )

                del volume_kspace, smaps_cpu

    return None

if __name__ == "__main__":
    # Iterate through the directories specified
    # Grab a sample k-space volume shaped (n_slices, n_coils, height, width)
    dirs = [ARGS.train, ARGS.val, ARGS.test]
    target_dir = ARGS.target
    method = ARGS.method
    print(dirs)
    print(target_dir)
    main(dirs, target_dir, method)
        
    
