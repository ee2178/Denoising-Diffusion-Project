import torch
from typing import Union, Tuple


def mask_uniform_subsample(
    m_omega: torch.Tensor,
    rho: Union[float, Tuple[float, float]] = 0.5,
) -> torch.Tensor:
    """
    Uniformly subsample a k-space mask.

    Args:
        m_omega: Boolean mask of shape (B, 1, Nx, Ny)
        rho:     Subsampling fraction or (min, max) range for random fraction

    Returns:
        m_lambda: Subsampled boolean mask, same shape as m_omega
    """
    if isinstance(rho, (int, float)):
        rho = (rho, rho)

    B, _, Nx, Ny = m_omega.shape
    m_lambda = torch.zeros_like(m_omega, dtype=torch.bool)

    for b in range(B):
        omega_indices = m_omega[b, 0].nonzero(as_tuple=False)  # (N, 2)
        N = omega_indices.shape[0]

        rho_b = rho[0] + (rho[1] - rho[0]) * torch.rand(1).item()
        N_rho = round(N * rho_b)

        perm = torch.randperm(N)[:N_rho]
        selected = omega_indices[perm]  # (N_rho, 2)

        m_lambda[b, 0, selected[:, 0], selected[:, 1]] = True

    return m_lambda

def mask_uniform_subsample_1D(
    m_omega: torch.Tensor,
    rho: Union[float, Tuple[float, float]] = 0.5,
    dim: int = 3,
) -> torch.Tensor:
    """
    Uniformly subsample a k-space mask along a single dimension (line subsampling).

    Args:
        m_omega: Boolean mask of shape (B, 1, Nx, Ny)
        rho:     Subsampling fraction or (min, max) range for random fraction
        dim:     Dimension along which to subsample lines (2 for Nx, 3 for Ny).
                 The mask is collapsed across all other spatial dims to find
                 acquired lines, then entire lines are kept or dropped together.

    Returns:
        m_lambda: Subsampled boolean mask, same shape as m_omega
    """
    if isinstance(rho, (int, float)):
        rho = (rho, rho)

    assert dim in (2, 3), f"dim must be 2 (Nx) or 3 (Ny), got {dim}"

    B = m_omega.shape[0]
    collapse_dim = 3 if dim == 2 else 2  # the spatial dim we reduce over

    m_lambda = torch.zeros_like(m_omega, dtype=torch.bool)

    for b in range(B):
        # A line is "acquired" if any point along the orthogonal dimension is acquired
        acquired_lines = m_omega[b, 0].any(dim=collapse_dim - 2)  # (Nx,) or (Ny,)
        omega_line_indices = acquired_lines.nonzero(as_tuple=False).squeeze(1)  # (N,)
        N = omega_line_indices.shape[0]

        rho_b = rho[0] + (rho[1] - rho[0]) * torch.rand(1).item()
        N_rho = round(N * rho_b)

        perm = torch.randperm(N)[:N_rho]
        selected = omega_line_indices[perm]  # (N_rho,)

        if dim == 2:
            m_lambda[b, 0, selected, :] = True
        else:
            m_lambda[b, 0, :, selected] = True

        # Mask back down to only acquired points in Ω
        m_lambda[b] = m_lambda[b] & m_omega[b]

    return m_lambda
    

def mask_gaussian_subsample(
    m_omega: torch.Tensor,
    rho: Union[float, Tuple[float, float]] = 0.5,
    sigma: float = 0.5,
) -> torch.Tensor:
    """
    Gaussian-weighted subsample a k-space mask, biased toward k-space center.

    Args:
        m_omega: Boolean mask of shape (B, 1, Nx, Ny)
        rho:     Subsampling fraction or (min, max) range for random fraction
        sigma:   Std dev of Gaussian as a fraction of half-FOV

    Returns:
        m_lambda: Subsampled boolean mask, same shape as m_omega
    """
    if isinstance(rho, (int, float)):
        rho = (rho, rho)

    B, _, Nx, Ny = m_omega.shape
    Cx, Cy = Nx // 2, Ny // 2
    device = m_omega.device

    m_lambda = torch.zeros_like(m_omega, dtype=torch.bool)

    for b in range(B):
        rho_b = rho[0] + (rho[1] - rho[0]) * torch.rand(1).item()
        N = m_omega[b, 0].sum().item()
        N_rho = round(N * rho_b)

        count = 0
        while count < N_rho:
            n_candidates = max((N_rho - count) * 4, 64)
            xs = (Cx + sigma * Cx * torch.randn(n_candidates, device=device)).round().long()
            ys = (Cy + sigma * Cy * torch.randn(n_candidates, device=device)).round().long()

            for x, y in zip(xs.tolist(), ys.tolist()):
                if count >= N_rho:
                    break
                if (
                    0 <= x < Nx
                    and 0 <= y < Ny
                    and m_omega[b, 0, x, y]
                    and not m_lambda[b, 0, x, y]
                ):
                    m_lambda[b, 0, x, y] = True
                    count += 1

    return m_lambda


def mask_subsample(
    m: torch.Tensor,
    rho: Union[float, Tuple[float, float]],
    type: str = "gaussian",
    **kwargs,
) -> torch.Tensor:
    """
    Subsample a k-space mask using the specified strategy.

    Args:
        m:    Boolean mask of shape (B, 1, Nx, Ny)
        rho:  Subsampling fraction or (min, max) range
        type: "gaussian" or "uniform"

    Returns:
        Subsampled boolean mask
    """
    if type == "uniform":
        return mask_uniform_subsample(m, rho)
    elif type == "gaussian":
        return mask_gaussian_subsample(m, rho, **kwargs)
    elif type == "uniform_1D":
        return mask_uniform_subsample_1D(m, rho, **kwargs)
    else:
        raise ValueError(f'mask_subsample type "{type}" not implemented.')


def ssdu_mask_subsample(
    m_omega: torch.Tensor,
    rho: Union[float, Tuple[float, float]],
    acs_size: int = 0,
    type: str = "gaussian",
    **kwargs,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Split an acquired k-space mask (Ω) into input (Λ) and loss (Ξ) masks for SSDU.

    Ω = Λ ∪ Ξ  (approximately, before optional ACS inclusion)

    Args:
        m_omega:  Boolean mask of shape (B, 1, Nx, Ny)
        rho:      Fraction of Ω points assigned to Λ (input mask)
        acs_size: Size of auto-calibration signal (ACS) region to include in both masks
        type:     Subsampling strategy — "gaussian" or "uniform"

    Returns:
        (m_lambda, m_xi): Input and loss boolean masks, same shape as m_omega
    """
    m_lambda = mask_subsample(m_omega, rho, type=type, **kwargs)
    m_xi = ~m_lambda & m_omega

    if acs_size > 0:
        _, _, Nx, Ny = m_omega.shape
        Cx, Cy = Nx // 2, Ny // 2
        acs_mask = torch.zeros_like(m_omega, dtype=torch.bool)

        if Cx > 0 and Cy > 0:
            acs_mask[:, :, Cx - acs_size // 2:Cx + acs_size // 2,
                          Cy - acs_size // 2:Cy + acs_size // 2] = True
        elif Cx > 0:
            acs_mask[:, :, Cx - acs_size // 2:Cx + acs_size // 2, :] = True
        elif Cy > 0:
            acs_mask[:, :, :, Cy - acs_size // 2:Cy + acs_size // 2] = True

        m_lambda = m_lambda | acs_mask
        m_xi = m_xi | acs_mask

    return m_lambda, m_xi
