import torch
import torch.nn.functional as F
import numpy as np
import os
import h5py
import argparse
parser = argparse.ArgumentParser()
parser.add_argument("--train", type=str, help="Run preprocessing over specified training set (provided path to image dir).", default=None)
parser.add_argument("--val", type=str, help="Run preprocessing over specified validation set (provided path to image dir).", default=None)
parser.add_argument("--test", type=str, help="Run preprocessing over specified test set (provided path to image dir).", default=None)
parser.add_argument("--target", type=str, help="Store processed images in a new target directory.", default=None)
ARGS = parser.parse_args()

# ChatGPT version of Nikola's espirit code 
def espirit(kspace: torch.Tensor, 
            acs_size=(32, 32), 
            kernel_size=8, 
            thresh_rowspace=0.05, 
            thresh_eig=0.95, 
            power_method=False, 
            rtol=1e-3, 
            maxit=100):
    """
    PyTorch version of ESPIRiT coil sensitivity estimation.

    Args:
        kspace: Tensor of shape (Nx, Ny, Ncoils, Batch)
        acs_size: Autocalibration region size
        kernel_size: Convolution kernel size
        thresh_rowspace: SVD threshold for rowspace
        thresh_eig: Eigenvalue threshold for sensitivity mask
        power_method: Not implemented
        rtol: Relative tolerance for power method
        maxit: Max iterations for power method

    Returns:
        smaps: Sensitivity maps (Nx, Ny, Ncoils, Batch) --> batch will be the size of the image volume
    """
    device = kspace.device
    Nx, Ny, Ncoils, B = kspace.shape
    Cx, Cy = Nx // 2, Ny // 2

    # 1. Extract ACS region
    x1 = slice(Cx - acs_size[0]//2, Cx + acs_size[0]//2)
    y1 = slice(Cy - acs_size[1]//2, Cy + acs_size[1]//2)
    kspace_acs = kspace[x1, y1, :, :]  # [acs_x, acs_y, Ncoils, B]

    # 2. Extract sliding patches: [num_patches, Ncoils * kernel_size**2, B]
    patches = kspace_acs.permute(3, 2, 0, 1)  # [B, C, H, W]
    unfold = torch.nn.Unfold(kernel_size=kernel_size)
    A = unfold(patches.view(B * Ncoils, 1, *kspace_acs.shape[:2]))  # [B*C, K², num_patches]
    A = A.view(B, Ncoils, kernel_size**2, -1).permute(3, 2, 1, 0).reshape(-1, kernel_size**2 * Ncoils, B)

    # 3. SVD
    if A.shape[0] < A.shape[1]:
        A_perm = A.permute(1, 0, 2)
        U, S, Vh = torch.linalg.svd(A_perm, full_matrices=False)
        V = U.conj()
    else:
        U, S, Vh = torch.linalg.svd(A, full_matrices=False)
        V = Vh

    # 4. Truncate basis using rowspace threshold
    vNbasis = [torch.sum(S[:, b] >= thresh_rowspace * S[0, b]).item() for b in range(B)]
    Nbasis = max(vNbasis)
    Vr = V[:, :Nbasis, :].clone()
    for b, nb in enumerate(vNbasis):
        Vr[:, nb:, b] = 0

    # 5. Reshape to k-space kernels
    Vkernel = Vr.view(kernel_size, kernel_size, Ncoils, Nbasis, B)

    # 6. Pad and FFT to image domain
    pad_x = (Nx - kernel_size) // 2
    pad_y = (Ny - kernel_size) // 2
    padded = F.pad(Vkernel.permute(4, 3, 2, 0, 1), (pad_y, pad_y, pad_x, pad_x))  # [B, Nbasis, C, H, W]
    Vk_imspace = torch.fft.ifft2(padded, dim=(-2, -1), norm='forward')  # Image domain

    # 7. Reshape for projection operator
    Vk_imspace = Vk_imspace.permute(3, 4, 2, 1, 0)  # [H, W, C, Nbasis, B]
    Vk_flat = Vk_imspace.view(Nx*Ny, Ncoils, Nbasis, B).permute(1, 2, 0, 3).reshape(Ncoils, Nbasis, Nx*Ny*B)

    # 8. Compute projection matrix Veff = V * Vᴴ
    Vh = Vk_flat.conj().permute(1, 0, 2)  # [Nbasis, Ncoils, ...]
    Veff = torch.matmul(Vk_flat, Vh) / (kernel_size ** 2)  # [C, C, ...]

    # 9. Eigen-decomposition
    if power_method:
        raise NotImplementedError("Power method is not implemented.")
    else:
        Q, L, _ = torch.linalg.svd(Veff, full_matrices=False)
        Q = Q[:, 0, :].reshape(Ncoils, Nx, Ny, B)
        L = L[0, :].reshape(Nx, Ny, 1, B)

    # 10. Mask and normalize phase
    mask = L > thresh_eig
    smaps = mask * Q.permute(1, 2, 0, 3).conj()
    smaps *= smaps[:, :, :1, :].conj() / (smaps[:, :, :1, :].abs() + 1e-8)

    return smaps

def main(dirs):
	for dir in dirs:
		if dir: 
			for name in os.listdir(dir):
				if name.startswith('')
				# Every file in these directories should be h5 files anyway
				hf = h5py.File(name)
				volume_kspace = hf['kspace'][()]
				# Convert to pytorch tensor (complex valued)
				volume_kspace = torch.from_numpy(volume_kspace)
				# Reshape operation (n_slices, n_coils, height, width) -> (Nx, Ny, Ncoils, Batch)
				volume_kspace = torch.reshape(volume_kspace, shape = (2, 3, 1, 0))
				smaps = espirit(volume_kspace)
				# Apply sensitivity maps and then sum
				volume_combined = torch.einsum('ijkl,ijkl->ijl', volume_kspace, smaps)
				# Convert to image domain (complex valued) after reshaping to change batch to first dimension
				volume_img = torch.fft.ifft2(torch.reshape(volume_combined, (2, 0, 1)))
    			# Save image domain volume
	return None

if __name__ == "__main__":
    # Iterate through the directories specified
    # Grab a sample k-space volume shaped (n_slices, n_coils, height, width)
    dirs = [ARGS.train, ARGS.val, ARGS.test]
    target_dir = ARGS.target
    main(dirs, target_dir)
        
    