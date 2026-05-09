"""SIGReg: Standard-Isotropic-Gaussian Regularizer.

Random unit-norm projections of the embeddings are tested against N(0, 1) using
the Epps-Pulley statistic (Gaussian-weighted L2 distance between the empirical
characteristic function and the standard-normal characteristic function).

By the Cramer-Wold theorem, agreement of every 1D marginal implies agreement of
the joint distribution, so averaging this statistic over many random directions
regularizes the embeddings towards isotropic standard Gaussian.

For a sample h_1, ..., h_N (one 1D projection):
    phi_N(t) = (1/N) sum_n exp(i t h_n)              (empirical char fn)
    phi_0(t) = exp(-t^2 / 2)                         (char fn of N(0, 1))
    T = integral w(t) |phi_N(t) - phi_0(t)|^2 dt
with w(t) = exp(-t^2 / 2) (bandwidth = 1) and trapezoidal quadrature on [t_min, t_max].
"""
from __future__ import annotations

import math
import torch


def _epps_pulley_1d(h: torch.Tensor, t_grid: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    """h: (..., N) projections; t_grid: (K,); weight: (K,) — quadrature weights * w(t).

    Returns the Epps-Pulley statistic per leading batch (...). Differentiable in h.
    """
    # th: (..., N, K) = h[..., None] * t_grid[None, :]
    th = h.unsqueeze(-1) * t_grid  # broadcasts to (..., N, K)
    cos_th = torch.cos(th).mean(dim=-2)  # (..., K) — Re[phi_N]
    sin_th = torch.sin(th).mean(dim=-2)  # (..., K) — Im[phi_N]
    re_phi0 = torch.exp(-0.5 * t_grid * t_grid)  # (K,)
    diff_sq = (cos_th - re_phi0).pow(2) + sin_th.pow(2)  # (..., K)
    # Integrate against w(t); weight already folds in trapezoid step + w(t).
    return (diff_sq * weight).sum(dim=-1)


def sigreg_loss(
    z: torch.Tensor,
    num_projections: int = 1024,
    t_min: float = 0.2,
    t_max: float = 4.0,
    num_quadrature: int = 32,
    bandwidth: float = 1.0,
) -> torch.Tensor:
    """Compute SIGReg(z).

    Args:
        z: (..., N, D) — N samples of D-dim embeddings (e.g. (T, B, D) or (B, D)).
        num_projections: M random unit-norm directions to test.
        t_min, t_max: trapezoidal quadrature range.
        num_quadrature: number of nodes.
        bandwidth: lambda in w(t) = exp(-t^2 / (2 lambda^2)). Default 1.

    Returns:
        Scalar tensor: mean over the M projections (and any leading sample-batch dims).
    """
    if z.dim() < 2:
        raise ValueError(f"z must have shape (..., N, D); got {tuple(z.shape)}")
    *lead, n_samples, d = z.shape
    if n_samples < 2:
        # Statistic is degenerate with <2 samples; return zero tensor connected to z's grad
        return z.sum() * 0.0

    # Random unit-norm directions on S^{d-1}, freshly drawn each call.
    u = torch.randn(num_projections, d, device=z.device, dtype=z.dtype)
    u = u / u.norm(dim=-1, keepdim=True).clamp_min(1e-8)

    # Project: (..., N, D) @ (D, M) -> (..., N, M); transpose to (..., M, N) for vectorized 1D test.
    h = z @ u.t()
    h = h.transpose(-1, -2)  # (..., M, N)

    t_grid = torch.linspace(t_min, t_max, num_quadrature, device=z.device, dtype=z.dtype)
    # Trapezoid step weights: dt at each interior node, dt/2 at endpoints.
    dt = (t_max - t_min) / (num_quadrature - 1)
    quad = torch.full_like(t_grid, dt)
    quad[0] = dt / 2
    quad[-1] = dt / 2
    w_t = torch.exp(-0.5 * (t_grid / bandwidth).pow(2))
    weight = quad * w_t  # (K,)

    stats = _epps_pulley_1d(h, t_grid, weight)  # (..., M)
    return stats.mean()
