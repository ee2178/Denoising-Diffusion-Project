import torch 
import torch.fft as fft
import torch.nn.functional as F
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
    y_coils = (fft.fft2(x_coils, norm = 'ortho'))
    # y_coils is C x N x N
    mask = acceleration_map
    y_mask = y_coils * mask[None]
    return y_mask


def mri_decoding(y, acceleration_map, smaps):
    # Apply mask to each channel of y
    y_mask = y * acceleration_map[None]
    # Apply ifft2
    x_coils = fft.ifft2(y_mask, norm = 'ortho')
    # Coil combination
    x = torch.einsum("ijk, ijk -> jk", smaps.conj(), x_coils)
    return x


def check_adjoint(E, EH, smaps):
    x = torch.randn(smaps.shape[1], smaps.shape[2],  dtype = torch.cfloat)
    x = x.to(smaps.device)
    y = torch.randn_like(E(x))

    # Check inner product <x, EH(y)>, <E(x), y>
    ip1 = torch.sum(x.conj() * EH(y))
    ip2 = torch.sum(E(x).conj() * y)

    diff = ip1 - ip2
    return diff, ip1, ip2

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


def make_acc_mask(
    shape,
    accel,
    acs_lines=24,
    seed=None,
    variable_density=False,
    dim=1
):
    """
    Create a diagonal undersampling matrix D for MRI.
    D is diag(mask.flatten()) so that D @ image_vec applies undersampling.
    Parameters
    ----------
    shape : tuple (Ny, Nx)
        k-space shape.
    accel : float
        Acceleration factor.
    acs_lines : int
        Number of central fully-sampled lines.
    seed : int or None
        RNG seed.
    variable_density : bool
        Use variable-density random sampling.
    dim : int
        Dimension to undersample (default = 1).
    Returns
    -------
    D : torch.sparse_coo_tensor
        A diagonal sparse matrix such that D @ vec(image) applies the mask.
    mask : torch.Tensor
        The 2D binary mask (Ny, Nx).
    """
    Ny, Nx = shape
    N = shape[dim]

    # Build empty mask
    mask = torch.zeros((Ny, Nx), dtype=torch.float32)
    
    # Determine how many samples to keep
    n_total = N
    n_acs = acs_lines
    n_outer = n_total - n_acs
    n_keep_outer = math.floor(n_outer / accel)

    # Build PDF
    if variable_density:
        x = torch.linspace(-1, 1, N)
        pdf = torch.exp(-4 * x**2)
        pdf = pdf / pdf.sum()
    else:
        pdf = torch.ones(N) / N

    # Remove ACS region from PDF
    center = N // 2
    half_acs = n_acs // 2
    pdf[center - half_acs : center + half_acs] = 0
    pdf = pdf / pdf.sum()

    if seed is not None:
        torch.manual_seed(seed)

    # Draw outer samples
    idx_outer = torch.multinomial(pdf, n_keep_outer, replacement=False)
    idx_acs = torch.arange(center - half_acs, center + half_acs)
    idx_keep = torch.cat([idx_outer, idx_acs]).unique()
    
    # Fill mask
    mask.index_fill_(dim, idx_keep, 1.0)

    # ---- Create dense diagonal matrix ----
    flat = mask.flatten()     # length Npix
    
    # From flat, generate a matrix that we can use as a matrix multiply
    D = torch.diag(flat[0:N])      # <-- full dense matrix
    
    return D, mask

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

    # Handle unfolding for complex tensors
    unfolded_real = F.unfold(y.real, kernel_size=(ks, ks), stride=stride)
    unfolded_imag = F.unfold(y.imag, kernel_size=(ks, ks), stride=stride)
    unfolded = torch.complex(unfolded_real, unfolded_imag)  # (B, C*ks*ks, Npatch)
    Npatch = unfolded.shape[-1]

    # (B, Npatch, C, ks*ks)
    Yp = unfolded.view(B, C, ks*ks, Npatch).permute(0, 3, 2, 1)  # (B, Npatch, ks*ks, C)

    # Covariance per patch
    X = torch.matmul(Yp.transpose(-1, -2).conj(), Yp)  # (B, Npatch, C, C)

    # SVD
    U, S, Vh = torch.linalg.svd(X, full_matrices=False)
    Q = U[..., 0]  # (B, Npatch, C)

    # Reference coil alignment per batch
    power = y.abs().pow(2).sum(dim=(2, 3))  # (B, C)
    Cref = power.argmax(dim=1)
    for b in range(B):
        ref = Q[b, :, Cref[b]]
        Q[b] *= ref.conj().sgn().unsqueeze(-1)

    # Reshape to low-res maps
    Hp = (H - ks) // stride + 1
    Wp = (W - ks) // stride + 1
    smaps_p = Q.permute(0, 2, 1).reshape(B, C, Hp, Wp)

    # Upsample to full size
    smaps_real = F.interpolate(smaps_p.real, size=(H, W), mode='bilinear', align_corners=False)
    smaps_imag = F.interpolate(smaps_p.imag, size=(H, W), mode='bilinear', align_corners=False)
    smaps = torch.complex(smaps_real, smaps_imag)

    # Normalize
    norm = smaps.abs().pow(2).sum(dim=1, keepdim=True)
    smaps /= (norm.sqrt() + 1e-8)
    return smaps
