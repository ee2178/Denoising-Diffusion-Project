"""
PyTorch translation of Julia noise-level estimation and multi-coil whitening utilities.

Tensor layout convention (matching the original Julia code):
    (H, W, C, B)  —  spatial height, spatial width, coils/channels, batch

CDF 9/7 wavelet coefficients are used as a default high-pass filter for
noise estimation (same as NNlib default in the Julia original).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Optional, Union, List, Tuple


# ---------------------------------------------------------------------------
# CDF 9/7 high-pass wavelet filter
# ---------------------------------------------------------------------------

cdf97 = torch.tensor(
    [
         0.091271763114,
        -0.057543526229,
        -0.591271763114,
         1.11508705,
        -0.591271763114,
        -0.057543526229,
         0.091271763114,
    ],
    dtype=torch.float64,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _real_dtype(x: Tensor) -> torch.dtype:
    """Return the underlying real dtype of x (handles complex tensors)."""
    return x.real.dtype if x.is_complex() else x.dtype


def _eps(x: Tensor) -> float:
    return torch.finfo(_real_dtype(x)).eps


def _default_filter(x: Tensor) -> Tensor:
    """Return cdf97 cast to the real dtype of x and on the same device."""
    return cdf97.to(device=x.device, dtype=_real_dtype(x))


def _separable_conv(x: Tensor, f: Tensor) -> Tensor:
    """
    Apply a separable 2-D convolution (along H then W) with 1-D filter *f*.

    Parameters
    ----------
    x : (H, W, C, B)  – real or complex
    f : (k,)           – real 1-D filter

    Returns
    -------
    Tensor, same shape as x.

    Notes
    -----
    Implemented as two depthwise 1-D convolutions so that each (channel, batch)
    slice is filtered independently, matching Julia's NNlib.conv behaviour.
    "Same" padding is used to preserve spatial dimensions.
    """
    H, W, C, B = x.shape
    k   = f.shape[0]
    pad = k // 2

    # ---- reshape to (1, B*C, H, W) for grouped / depthwise conv2d ----------
    x_pt = x.permute(3, 2, 0, 1).reshape(1, B * C, H, W)   # (1, B*C, H, W)

    f_ = f.to(dtype=_real_dtype(x_pt))
    fh = f_.reshape(1, 1, k, 1).expand(B * C, 1, k, 1).contiguous()
    fw = f_.reshape(1, 1, 1, k).expand(B * C, 1, 1, k).contiguous()

    def _conv_real(t: Tensor) -> Tensor:
        t = F.conv2d(t, fh, padding=(pad, 0), groups=B * C)
        t = F.conv2d(t, fw, padding=(0, pad), groups=B * C)
        return t[..., :H, :W]          # trim any extra samples from odd padding

    if x_pt.is_complex():
        out = torch.complex(_conv_real(x_pt.real), _conv_real(x_pt.imag))
    else:
        out = _conv_real(x_pt)

    return out.reshape(B, C, H, W).permute(2, 3, 1, 0)      # (H, W, C, B)


def _mul_channel(M: Tensor, t: Tensor) -> Tensor:
    """
    Apply a batched (C×C) matrix to the channel dimension of *t*.

    Equivalent to Julia's ``mul_channel(M, t)`` / the ⊠ batched_adjoint pattern.

    Parameters
    ----------
    M : (C, C, B)
    t : (H, W, C, B)

    Returns
    -------
    Tensor, shape (H, W, C, B)
    """
    H, W, C, B = t.shape
    M_b = M.permute(2, 0, 1)                    # (B, C, C)
    t_b = t.permute(3, 2, 0, 1).reshape(B, C, H * W)  # (B, C, H*W)
    out = torch.bmm(M_b, t_b)                   # (B, C, H*W)
    return out.reshape(B, C, H, W).permute(2, 3, 1, 0)   # (H, W, C, B)


def _batched_adjoint(M: Tensor) -> Tensor:
    """
    Conjugate-transpose the first two dimensions of a (C, C, B) tensor,
    mirroring Julia's ``NNlib.batched_adjoint``.
    """
    return M.conj().permute(1, 0, 2)


# ---------------------------------------------------------------------------
# Noise-Level Estimation  (MAD)
# ---------------------------------------------------------------------------

def nle_mad(
    x: Tensor,
    f: Optional[Tensor] = None,
) -> Tensor:
    """
    Median Absolute Deviation noise-level estimator.

    Parameters
    ----------
    x : Tensor, shape (H, W, C, B) – real **or** complex multicoil signal.
    f : Tensor, optional
        1-D high-pass filter.  Defaults to the CDF 9/7 filter ``cdf97``.

    Returns
    -------
    Tensor  – scalar estimate of the noise standard deviation.

    Notes
    -----
    For complex input the real and imaginary parts are stacked along the
    channel axis and the result is scaled by √2, giving the std-dev of a
    circularly-symmetric complex normal distribution (matching the Julia
    ``Complex`` method).
    """
    if f is None:
        f = _default_filter(x)

    if x.is_complex():
        y = torch.cat([x.real, x.imag], dim=2)   # (H, W, 2C, B)
        return nle_mad(y, f) * (2.0 ** 0.5)

    z = _separable_conv(x, f)
    # Divide by 2: the 1-D filter is applied twice (H and W directions)
    return torch.median(z.abs()) / (2.0 * 0.6745)


# ---------------------------------------------------------------------------
# Noise Covariance Estimation
# ---------------------------------------------------------------------------

def ncov_est(
    x: Tensor,
    f: Optional[Tensor] = None,
) -> Tensor:
    """
    Estimate the inter-channel noise covariance matrix.

    Parameters
    ----------
    x : Tensor, shape (H, W, C, B)
    f : Tensor, optional – 1-D filter (default: ``cdf97``).

    Returns
    -------
    Tensor, shape (C, C, B)
    """
    if f is None:
        f = _default_filter(x)

    H, W, C, B = x.shape

    # Filter every (channel × batch) slice independently
    xr  = x.reshape(H, W, 1, C * B)
    zr  = _separable_conv(xr, f) / 2.0      # (H, W, 1, C*B)

    Z   = zr.reshape(-1, C, B)              # (N, C, B),  N = H*W
    N   = Z.shape[0]
    Z   = Z.permute(1, 0, 2)               # (C, N, B)

    mu  = Z.mean(dim=1, keepdim=True)       # (C, 1, B)
    Zmu = (Z - mu).permute(2, 0, 1).reshape(N * B, C, 1)   # (N*B, C, 1)

    # Accumulate outer products: Σ_n  zμ · zμᴴ
    Sigma = torch.bmm(Zmu, Zmu.conj().transpose(-2, -1))    # (N*B, C, C)
    Sigma = Sigma.reshape(N, B, C, C).sum(dim=0)            # (B, C, C)

    # Normalise by the number of non-zero (masked) pixels
    mask   = (x.abs().pow(2).sum(dim=2) > 0)               # (H, W, B)
    N_mask = mask.sum(dim=(0, 1)).reshape(B, 1, 1).to(Sigma.dtype)

    Sigma  = Sigma / N_mask                                  # (B, C, C)
    return Sigma.permute(1, 2, 0)                           # (C, C, B)


# ---------------------------------------------------------------------------
# Covariance Matrix Square Root
# ---------------------------------------------------------------------------

def sqrt_covmat(Sigma: Tensor) -> Tensor:
    """
    Matrix square root of a positive semi-definite matrix (or batch thereof).

    For a PSD matrix Σ = U S Uᴴ the square root is  √Σ = U √S Uᴴ.

    Parameters
    ----------
    Sigma : Tensor, shape (C, C) or (C, C, B)

    Returns
    -------
    Tensor, same shape as *Sigma*.
    """
    if Sigma.dim() == 2:
        U, s, _ = torch.linalg.svd(Sigma)
        # √Σ = U · diag(√s) · Uᴴ
        return U @ (s.sqrt().unsqueeze(0) * U.conj().T)

    # Batched: Sigma is (C, C, B)
    Sig_b           = Sigma.permute(2, 0, 1)          # (B, C, C)
    U, s, _         = torch.linalg.svd(Sig_b)         # U:(B,C,C),  s:(B,C)
    sqrt_s          = s.sqrt().unsqueeze(1)            # (B, 1, C)
    result          = U @ (sqrt_s * U.conj().transpose(-2, -1))  # (B, C, C)
    return result.permute(1, 2, 0)                     # (C, C, B)


# ---------------------------------------------------------------------------
# Whitening
# ---------------------------------------------------------------------------

def _inv_sqrt_sigma(
    U:   Tensor,   # (C, C, B) – eigenvectors of Σ
    s:   Tensor,   # (C, B)    – eigenvalues  of Σ
    t:   Tensor,   # (H, W, C, B)
    eps: float,
) -> Tensor:
    """Apply Σ^{-1/2} = U diag(1/√s) Uᴴ channel-wise to *t*."""
    t1          = _mul_channel(_batched_adjoint(U), t)       # Uᴴ t
    inv_sqrt_s  = 1.0 / (s.sqrt() + eps)                    # (C, B)
    t2          = t1 * inv_sqrt_s.unsqueeze(0).unsqueeze(0) # (H, W, C, B)
    return _mul_channel(U, t2)                               # U t2


def _coil_combine(smaps: Tensor, data: Tensor) -> Tensor:
    """Sensitivity-weighted coil combination → (H, W, 1, B)."""
    return (smaps.conj() * data).sum(dim=2, keepdim=True)


def whiten(
    x:      Union[Tensor, List[Tensor]],
    smaps:  Optional[Tensor] = None,
    Sigma:  Optional[Tensor] = None,
) -> Union[Tensor, dict]:
    """
    Whiten multicoil image-domain data.

    Three calling conventions (matching the Julia overloads):

    1. ``whiten(x)``  or  ``whiten(x, Sigma=...)``
       No sensitivity maps: returns the whitened data tensor directly.

    2. ``whiten(x, smaps)``  or  ``whiten(x, smaps, Sigma)``
       Single data tensor with sensitivity maps.

    3. ``whiten([x1, x2, ...], smaps)``  or  ``whiten([...], smaps, Sigma)``
       List of data tensors sharing the same sensitivity maps.

    Parameters
    ----------
    x     : Tensor (H, W, C, B)  **or** list of such tensors.
    smaps : Tensor (H, W, C, B), optional – sensitivity maps.
    Sigma : Tensor (C, C, B),    optional – noise covariance.
            Estimated from *x* (or *x[0]*) via :func:`ncov_est` if omitted.

    Returns
    -------
    If *smaps* is ``None``
        Tensor – whitened data, shape (H, W, C, B).
    Otherwise
        dict with keys:
        - ``"data"``  – whitened data (Tensor or list of Tensors)
        - ``"smaps"`` – whitened & normalised sensitivity maps
        - ``"sigma"`` – per-pixel scale factor, shape (H, W, 1, B)
        - ``"zinv"``  – inverse normalisation map,  shape (H, W, 1, B)
    """

    # ── Case 1: no sensitivity maps ─────────────────────────────────────────
    if smaps is None:
        assert isinstance(x, Tensor), "smaps=None requires a single Tensor x"
        if Sigma is None:
            Sigma = ncov_est(x)
        U_b, s_b, _ = torch.linalg.svd(Sigma.permute(2, 0, 1))  # (B,C,C)
        U = U_b.permute(1, 2, 0)   # (C,C,B)
        s = s_b.permute(1, 0)      # (C,B)
        return _inv_sqrt_sigma(U, s, x, _eps(x))

    # ── Case 2 & 3: sensitivity maps present ────────────────────────────────
    xs: List[Tensor] = list(x) if isinstance(x, (list, tuple)) else [x]

    if Sigma is None:
        Sigma = ncov_est(xs[0])

    # SVD of Σ — work in (B, C, C) internally, then convert back
    U_b, s_b, _ = torch.linalg.svd(Sigma.permute(2, 0, 1))   # (B,C,C), (B,C)
    U = U_b.permute(1, 2, 0)   # (C,C,B)
    s = s_b.permute(1, 0)      # (C,B)
    eps = _eps(smaps)

    def sq_inv(t: Tensor) -> Tensor:
        return _inv_sqrt_sigma(U, s, t, eps)

    # Whiten data and smaps
    xs_w    = [sq_inv(xi) for xi in xs]
    smaps_w = sq_inv(smaps)

    # Normalise whitened sensitivity maps  ‖smap_w(h,w)‖ → 1
    z       = smaps_w.abs().pow(2).sum(dim=2, keepdim=True).sqrt()  # (H,W,1,B)
    smaps_w = smaps_w / (z + eps)

    # Re-scale so the dynamic range of the coil-combined whitened image
    # matches that of the original coil-combined image
    if len(xs) == 1:
        beta  = _coil_combine(smaps,   xs[0]  ).abs().amax(dim=(0, 1), keepdim=True)
        delta = _coil_combine(smaps_w, xs_w[0]).abs().amax(dim=(0, 1), keepdim=True)
    else:
        beta  = torch.stack(
            [_coil_combine(smaps,   xi  ).abs().amax(dim=(0, 1)) for xi in xs],    dim=0
        ).mean(0).unsqueeze(0)
        delta = torch.stack(
            [_coil_combine(smaps_w, xi_w).abs().amax(dim=(0, 1)) for xi_w in xs_w], dim=0
        ).mean(0).unsqueeze(0)

    sigma = beta / delta                                        # (1, 1, 1, B)
    xs_w  = [xi_w * sigma for xi_w in xs_w]

    # Renormalisation maps
    zinv  = (z > 0).to(sigma.dtype) / (sigma * z + eps)       # (H, W, 1, B)
    sigma = (z > 0).to(sigma.dtype) * sigma                    # (H, W, 1, B)

    result_data = xs_w[0] if not isinstance(x, (list, tuple)) else xs_w
    return dict(data=result_data, smaps=smaps_w, sigma=sigma, zinv=zinv)


def whiten_5d(
    y:     Tensor,
    smaps: Tensor,
    Sigma: Tensor,
) -> dict:
    """
    Whiten a 5-D data tensor by slicing along the last dimension.

    Parameters
    ----------
    y     : Tensor, shape (H, W, C, B, N)
    smaps : Tensor, shape (H, W, C, B)
    Sigma : Tensor, shape (C, C, B)

    Returns
    -------
    Same dict as :func:`whiten` but ``"data"`` is a 5-D Tensor.
    """
    slices = [y[..., ii] for ii in range(y.shape[-1])]
    result = whiten(slices, smaps, Sigma)
    result["data"] = torch.stack(result["data"], dim=-1)
    return result
