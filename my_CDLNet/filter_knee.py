# We need to filter out the PD scans from the PDFS scans over the smap and kspace knee scan directories. 
# This is necessary for the denoiser training since there is no filtration step. 

import os
import h5py
import shutil
from pathlib import Path

# -------------------------
# Paths
# -------------------------
processed_root = Path("/home/ee2178/scratch/ee2178/datasets/fastmri_preprocessed/knee_coil_combined")
kspace_root = Path("/home/ee2178/scratch/ee2178/datasets/fastmri/knee")

processed_dirs = {
    "train": processed_root / "train",
    "val": processed_root / "val",
    "test": processed_root / "test",
}

kspace_dirs = {
    "train": kspace_root / "multicoil_train",
    "val": kspace_root / "multicoil_val",
    "test": kspace_root / "multicoil_test_v2",
}

# Output folders
pd_root = processed_root / "pd"
pdfs_root = processed_root / "pdfs"

for split in ["train", "val", "test"]:
    (pd_root / split).mkdir(parents=True, exist_ok=True)
    (pdfs_root / split).mkdir(parents=True, exist_ok=True)


# -------------------------
# Helper function
# -------------------------
def get_acquisition_type(h5_file):
    """
    Returns 'PD', 'PDFS', or None
    """
    with h5py.File(h5_file, "r") as f:
        if "acquisition" in f.attrs:
            acq = f.attrs["acquisition"]
            if isinstance(acq, bytes):
                acq = acq.decode()
            return acq
    return None


# -------------------------
# Main loop
# -------------------------
for split in ["train", "val", "test"]:

    kspace_dir = kspace_dirs[split]
    processed_dir = processed_dirs[split]

    print(f"\nProcessing {split} set...")

    for kspace_file in kspace_dir.glob("*.h5"):

        acquisition = get_acquisition_type(kspace_file)

        if acquisition is None:
            print(f"Skipping {kspace_file.name} (no acquisition attr)")
            continue

        processed_file = processed_dir / kspace_file.name

        if not processed_file.exists():
            print(f"Missing processed file: {processed_file}")
            continue

        if "PDFS" in acquisition:
            dest = pdfs_root / split / processed_file.name
        elif "PD" in acquisition:
            dest = pd_root / split / processed_file.name
        else:
            print(f"Unknown acquisition {acquisition} for {kspace_file.name}")
            continue
        # Don't copy, make a symlink instead
        # shutil.copy2(processed_file, dest)
        os.symlink(processed_file, dest)

    print(f"Finished {split}")

