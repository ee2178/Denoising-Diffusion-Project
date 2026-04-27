import torch
import torch.fft as fft
import matplotlib.pyplot as plt
import json
import os
import h5py
from model import CDLNet, LPDSNet, LPDSNetBase
from PIL import Image
from torchvision.transforms.functional import to_tensor

def img_load(path, gray=False):
	""" Load batched tensor image (1,C,H,W) from file path.
	"""
	if gray:
		return to_tensor(Image.open(path).convert('L'))[None,...]
	return to_tensor(Image.open(path))[None,...]

def gen_bayer_mask(x):
    ''' 
    RGB --> [ R  G ]
            [ G  B ]
    '''    
    m = torch.zeros_like(x)
    m[:,0,0::2,0::2] = 1 # R
    m[:,1,0::2,1::2] = 1 # G1
    m[:,1,1::2,0::2] = 1 # G2
    m[:,2,1::2,1::2] = 1 # B
    return m

def awgn(input, noise_std):
	""" Additive White Gaussian Noise
	y: clean input image
	noise_std: (tuple) noise_std of batch size N is uniformly sampled 
	           between noise_std[0] and noise_std[1]. Expected to be in interval
			   [0,255]
	"""
	if not isinstance(noise_std, (list, tuple)):
		sigma = noise_std
	else: # uniform sampling of sigma
		sigma = noise_std[0] + \
		       (noise_std[1] - noise_std[0])*torch.rand(len(input),1,1,1, device=input.device)
	return input + torch.randn_like(input) * (sigma/255), sigma

def contrast_enhance(mag_array, thresh = 1.):
    # assume input is real valued
    # We are mainly going to use this to enhance the contrast of MR images
    # Clip values with magnitude greater than thresh
    return torch.clamp(mag_array, max = thresh)
def saveimg(x, name, contrast=True, thresh = 1.):
    # Helper function to images
    # get abs value
    x = torch.squeeze(x.abs()).detach().cpu()
    if contrast == True:
        x = contrast_enhance(x, thresh = thresh)
    plt.imshow(x, cmap = 'gray')
    plt.axis('off')
    plt.savefig(name, bbox_inches='tight', pad_inches = 0)
    print("Saved image to " + name + ".")

def load_data_knee( kspace_fname = "file1000031.h5",                                            # Path to kspace
                    slice = None,                                                               # Slice to extract
                    kspace_root = "../../datasets/fastmri/knee/multicoil_val",   # KSpace base path
                    smap_root = "../../datasets/fastmri_preprocessed/knee_coil_combined/pd/val",# Path to smaps
                    device = 'cpu',                                                             # Device
                    scale_fac = 5e3                                                             # KSpace scale factor
                    ):
    fname = os.path.join(kspace_root, kspace_fname)

    # Search in val dir for corresponding smaps
    smaps_fname = os.path.join(smap_root, kspace_fname)

    with h5py.File(smaps_fname) as f:
        x = f['image'][()]
        smaps = f['smaps'][()]
        gnd_truth = f['image'][()]

    with h5py.File(fname) as f:
        kspace = f['kspace'][()]
    kspace = torch.from_numpy(kspace)
    smaps = torch.from_numpy(smaps)
    smaps = torch.squeeze(smaps)
    # Send to GPU
    gnd_truth = torch.from_numpy(gnd_truth).to(device)*scale_fac
    smaps = smaps.to(device)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*scale_fac
    # Compute the organ mask by summing over absolute value of smaps in the coil dimension 
    mask = torch.sum(smaps.abs(), dim = 1, keepdim = True) > 0

    # If slice is not none, only return that slice
    if slice is not None:
        kspace = kspace[slice:slice+1, :, :, :]
        smaps = smaps[slice:slice+1, :, :, :]
        mask = mask[slice:slice+1, :, :, :]
        gnd_truth = gnd_truth[slice:slice+1, :, :]

    return kspace, smaps, mask, gnd_truth

def load_data_brain(kspace_fname = "file_brain_AXT2_200_2000572.h5",                              # Path to kspace
                    slice = None,                                                                 # Slice to extract
                    kspace_root = "../../datasets/fastmri/brain/multicoil_val",                   # KSpace base path
                    smap_root = "../../datasets/fastmri_preprocessed/brain_T2W_coil_combined/val",# Path to smaps
                    device = 'cpu',                                                               # Device
                    scale_fac = 2e3                                                               # KSpace scale factor
                    ):
    fname = os.path.join(kspace_root, kspace_fname)

    # Search in val dir for corresponding smaps
    smaps_fname = os.path.join(smap_root, kspace_fname)

    with h5py.File(smaps_fname) as f:
        x = f['image'][()]
        smaps = f['smaps'][()]
        gnd_truth = f['image'][()]

    with h5py.File(kspace_fname) as f:
        kspace = f['kspace'][()]
    kspace = torch.from_numpy(kspace)
    smaps = torch.from_numpy(smaps)
    smaps = torch.squeeze(smaps)
    # Send to GPU
    gnd_truth = torch.from_numpy(gnd_truth).to(device)*scale_fac
    smaps = smaps.to(device)
    # Scale kspace and send to GPU
    kspace = kspace.to(device)*scale_fac
    # Compute the organ mask by summing over absolute value of smaps in the coil dimension
    mask = torch.sum(smaps.abs(), dim = 1, keepdim = True) > 0

    # If slice is not none, only return that slice
    if slice is not None:
        kspace = kspace[slice:slice+1, :, :, :]
        smaps = smaps[slice:slice+1, :, :, :]
        mask = mask[slice:slice+1, :, :, :]
        gnd_truth = gnd_truth[slice:slice+1, :, :]

    return kspace, smaps, mask, gnd_truth

def make_coordinate_grid(H, W, device):
    y = torch.linspace(-1, 1, H, device=device)
    x = torch.linspace(-1, 1, W, device=device)
    yy, xx = torch.meshgrid(y, x, indexing='ij')
    return xx, yy  # shape (H, W)

def polynomial_basis(xx, yy, degree):
    """
    Returns list of basis terms up to given total degree.
    Example for degree=2:
    [1, x, y, x^2, xy, y^2]
    """
    basis = []
    for i in range(degree + 1):
        for j in range(degree + 1 - i):
            basis.append((xx ** i) * (yy ** j))
    return basis  # list of (H, W)

def complex_polynomial_field(
    shape,
    degree=3,
    temperature=0.1,
    centered=True
):
    """
    shape: (B, C, H, W)
    returns: complex tensor (B, C, H, W)
    """
    B, C, H, W = shape
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    xx, yy = make_coordinate_grid(H, W, device)
    basis = polynomial_basis(xx, yy, degree)
    K = len(basis)

    # stack basis → (K, H, W)
    basis_stack = torch.stack(basis, dim=0)

    # random coefficients per batch
    coeff_real = torch.randn(B, K, 1, 1, device=device)
    coeff_imag = torch.randn(B, K, 1, 1, device=device)

    # linear combination
    real = torch.sum(coeff_real * basis_stack.unsqueeze(0), dim=1, keepdim=True)
    imag = torch.sum(coeff_imag * basis_stack.unsqueeze(0), dim=1, keepdim=True)

    # normalize for stability
    def normalize(x):
        x = x - x.mean(dim=(-2, -1), keepdim=True)
        x = x / (x.std(dim=(-2, -1), keepdim=True) + 1e-8)
        return x

    real = normalize(real)
    imag = normalize(imag)

    field = real + 1j * imag  # (B,1,H,W)

    if centered:
        field = 1 + temperature * field
    else:
        field = temperature * field

    field = field.expand(-1, C, -1, -1)
    return field.to(torch.complex64)


def modulate_with_polynomial_field(
    x,
    degree=3,
    temperature=0.1,
    centered=True
):
    """
    x: complex tensor (B, C, H, W)
    """
    assert torch.is_complex(x)

    M = complex_polynomial_field(
        x.shape,
        degree=degree,
        temperature=temperature,
        centered=centered
    )

    return x * M