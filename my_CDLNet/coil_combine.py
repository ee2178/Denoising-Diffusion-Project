import os
import argparse
import h5py
import torch

from mri_utils import ifftc  # uses your existing centered IFFT

parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, default=None)
parser.add_argument("--val", type=str, default=None)
parser.add_argument("--test", type=str, default=None)
parser.add_argument("--target", type=str, required=True, help="Target dir containing H5s with 'smaps'.")
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--overwrite", action="store_true")
parser.add_argument("--no_norm", action="store_true", help="Skip division by sum |smaps|^2.")
ARGS = parser.parse_args()


def infer_split(dir_path: str) -> str:
    base = os.path.basename(os.path.normpath(dir_path))
    if base == "test_v2":
        return "test"
    if base in ("train", "val", "test"):
        return base
    if dir_path.endswith("train"):
        return "train"
    if dir_path.endswith("val"):
        return "val"
    if dir_path.endswith("test") or dir_path.endswith("test_v2"):
        return "test"
    return base


def coil_combine(img_coils, smaps, eps=1e-12, normalize=True):
    """
    SENSE-style coil combination.

    img_coils, smaps: (S, C, H, W) complex tensors
    returns: (S, H, W) complex tensor
    """
    num = (smaps.conj() * img_coils).sum(dim=1)
    if not normalize:
        return num
    denom = (smaps.abs() ** 2).sum(dim=1).clamp_min(eps)
    return num / denom


def main():
    if ARGS.device.startswith("cuda") and torch.cuda.is_available():
        device = torch.device(ARGS.device)
    else:
        device = torch.device("cpu")

    src_dirs = [ARGS.train, ARGS.val, ARGS.test]

    for src_dir in src_dirs:
        if not src_dir:
            continue

        split = infer_split(src_dir)
        tgt_split_dir = os.path.join(ARGS.target, split)
        os.makedirs(tgt_split_dir, exist_ok=True)

        for name in os.listdir(src_dir):
            if not name.startswith("file"):
                continue

            src_path = os.path.join(src_dir, name)
            tgt_path = os.path.join(tgt_split_dir, name)

            if not os.path.exists(tgt_path):
                print(f"[SKIP] target missing (no smaps file): {tgt_path}")
                continue

            # skip if already has image (unless overwrite)
            with h5py.File(tgt_path, "r") as tf:
                if "smaps" not in tf:
                    print(f"[SKIP] no smaps in target: {tgt_path}")
                    continue
                if ("image" in tf) and (not ARGS.overwrite):
                    print(f"[SKIP] image exists: {tgt_path}")
                    continue

            # load smaps from target
            with h5py.File(tgt_path, "r") as tf:
                smaps = torch.from_numpy(tf["smaps"][()])  # (S,C,H,W) complex

            # load kspace from source
            with h5py.File(src_path, "r") as sf:
                kspace = torch.from_numpy(sf["kspace"][()])  # (S,C,H,W) complex

            if kspace.shape != smaps.shape:
                print(f"[WARN] shape mismatch {name}: kspace {kspace.shape} vs smaps {smaps.shape}. Skipping.")
                continue

            print(f"[PROC] {split}/{name}  shape={kspace.shape}  device={device}")

            with torch.inference_mode():
                kspace_d = kspace.to(device, non_blocking=True)
                smaps_d  = smaps.to(device, non_blocking=True)

                # single centered IFFT over the whole volume
                img_coils = ifftc(kspace_d)  # (S,C,H,W)

                # coil combine
                combined = coil_combine(img_coils, smaps_d, normalize=(not ARGS.no_norm))  # (S,H,W)

                combined_np = combined.detach().cpu().numpy()

                del kspace_d, smaps_d, img_coils, combined

            # write image into target file
            with h5py.File(tgt_path, "a") as tf:
                if "image" in tf:
                    if ARGS.overwrite:
                        del tf["image"]
                    else:
                        print(f"[SKIP] image exists (race): {tgt_path}")
                        continue
                tf.create_dataset("image", data=combined_np)

            print(f"[DONE] wrote 'image' to {tgt_path}")

            del kspace, smaps

    return None


if __name__ == "__main__":
    main()
