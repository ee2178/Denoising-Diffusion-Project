import torch 
import torch.fft as fft
import math

# Let us write a helper function to give us a DFT matrix for some N
def dft_matrix(N):
    # Generate a vector [0, 1, ..., N-1]
    n = torch.linspace(0, N-1, N)
    # Generate meshgrid based on n
    Nx, Ny = torch.meshgrid(n, n)
    W = Nx * Ny
    # Turn our W matrix into a complex valued matrix with W as the imaginary components
    W = torch.complex(torch.zeros_like(W), -2*torch.Tensor([math.pi])*W/N)
    return torch.exp(W)

def mri_encoding(x, acceleration_map, smaps):
    # We take an acceleration_map to be a row-removed identity matrix corresponding to how many lines in kspace we keep [N x N]
    # We take a sensitivity map and assume it performs elementwise multiplication in the image domain [C x N x N]
    x_coils = smaps[:, :]* x[None, :, :]
    # x_coils is C x N x N
    y_coils = fft.fftshift(fft.fft2(x_coils, norm = 'ortho'))
    # y_coils is C x N x N
    y_mask = torch.einsum('jj, ijk -> ijk', acceleration_map, y_coils)
    return y_mask


def mri_decoding(y, acceleration_map, smaps):
    # Apply mask to each channel of y
    y_mask = torch.einsum('jj, ijk -> ijk', acceleration_map, y)
    # Apply ifft2
    x_coils = fft.fftshift(fft.ifft2(y_mask, norm = 'ortho'))
    # Coil combination
    x = torch.einsum("ijk, ijk -> jk", smaps.conj(), x_coils)
    return x


def detect_acc_mask(y):
    '''  
    This function takes some sample image and detects the acceleration map.
    Assume y is C x H x W, C the number of coils. 
    '''
    # Transpose to look at depleted columns, not rows
    y = y.permute(0, 2, 1)
    # Look along first column of coil image of y. Detect missing rows. 
    nonzeros = torch.sum(y[0, :, :] != 0.0 + 0.0j, dim = -1)
    mask = torch.diag((nonzeros > 0)*1)
    # Returns H x W mask 
    return mask
