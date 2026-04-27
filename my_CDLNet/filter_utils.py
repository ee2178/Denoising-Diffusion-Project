import os
import numpy as np
import torch
import train
import json
from torchvision.utils import make_grid, save_image

def complex_to_rgb_grid(W, nrow, scale_each=False, global_max=None):
    """
    Convert complex filters to RGB:
        R = real
        G = 0
        B = imag
    """
    real = torch.real(W)
    imag = torch.imag(W)

    if scale_each:
        rmax = real.abs().amax(dim=(1,2,3), keepdim=True) + 1e-8
        imax = imag.abs().amax(dim=(1,2,3), keepdim=True) + 1e-8
    else:
        rmax = global_max + 1e-8
        imax = global_max + 1e-8

    real = real / rmax
    imag = imag / imax

    # Normalize to [0,1]
    real = (real + 1) / 2
    imag = (imag + 1) / 2

    green = torch.zeros_like(real)

    rgb = torch.stack([real, green, imag], dim=2)  # (N, C, 3, H, W)
    rgb = rgb.squeeze(1)  # assume C=1 → (N, 3, H, W)

    grid = make_grid(rgb, nrow=nrow, padding=2)
    return grid


def get_B_complex(B_layer):
    """Reconstruct complex weights from ComplexConvTranspose2d"""
    return torch.complex(
        B_layer.conv_real.weight.detach(),
        B_layer.conv_imag.weight.detach()
    )


def filters_lpds(net, save_dir, scale_each=False):
    print("--------- LPDS filters ---------")
    save_dir = os.path.join(save_dir, "filters")
    os.makedirs(save_dir, exist_ok=True)

    assert hasattr(net, "A") and hasattr(net, "B")

    K = net.K

    # Collect filters
    AL, BL = [], []
    mmax = 0

    for k in range(K):
        A = net.A[k].weight.detach()
        B = get_B_complex(net.B[k])

        AL.append(A)
        BL.append(B)

        mmax = max(mmax, A.abs().max().item(), B.abs().max().item())

    # Grid size
    n = int(np.ceil(np.sqrt(AL[0].shape[0])))

    # Save A/B filters
    for k in range(K):
        Ag = complex_to_rgb_grid(AL[k], nrow=n, scale_each=scale_each, global_max=mmax)
        Bg = complex_to_rgb_grid(BL[k], nrow=n, scale_each=scale_each, global_max=mmax)

        fn = os.path.join(save_dir, f"AB{k:02d}_{scale_each}.png")
        print(f"Saving {fn} ...")

        save_image(torch.stack([Ag, Bg]), fn, nrow=2, padding=5)

    # Save dictionary D (same as B[0])
    D = get_B_complex(net.D)
    Dg = complex_to_rgb_grid(D, nrow=n, scale_each=scale_each, global_max=mmax)

    fn = os.path.join(save_dir, f"D_{scale_each}.png")
    print(f"Saving {fn} ...")
    save_image(Dg, fn)
