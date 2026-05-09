"""Diagnostic: does SIGReg actually penalize low-rank embeddings?

Hypothesis from the probing results: trained encoders show effective rank
~3-7 on D=320, yet sigreg stays near its trained value (~0.06-0.10). Two
explanations are possible:

  (a) The regularizer literally cannot tell rank-3 from full-rank when each
      individual dimension has been BN-normalized to unit variance. With M=1024
      random unit-norm directions on S^319, each direction projects 99% of its
      energy onto noise dimensions; the rank-3 signal contaminates only a
      tiny fraction of the directions. So all 1D marginals look ~N(0,1)
      and SIGReg is satisfied.

  (b) Our implementation has a bug. We can rule this out by feeding the
      regularizer a known-good full-rank Gaussian and checking the value is
      near zero, then perturbing it.

This script feeds sigreg_loss several controlled distributions:

  1. Full-rank N(0, I_D)                 -- expected sigreg ~ small
  2. Full-rank N(0, I_D) but with one    -- the obvious failure mode (a single
     coordinate scaled by 5x                non-unit marginal); should spike
  3. Rank-K signal in K random axes,     -- the actual failure mode of interest
     padded with N(0, 1) on the other       (BN-normalized so each dim still has
     D-K dims, then per-dim BN'd            unit variance, but information lives
                                            in K dims)
  4. Rank-K signal in K random axes,     -- variant: don't pad with noise.
     other D-K dims set to 0, then per-     This zero-pads then BN'd: the zeroed
     dim BN'd                               dims are degenerate, so per-dim BN
                                            keeps them noise after BN's epsilon.

We sweep K in {1, 3, 8, 32, 64, 128, 320} (320 = full rank).

Run:
  .venv/Scripts/python.exe -m diagnostics.sigreg_rank
"""
from __future__ import annotations

import argparse
import torch

from lewm.sigreg import sigreg_loss


def per_dim_bn(x: torch.Tensor, eps: float = 1e-5) -> torch.Tensor:
    mu = x.mean(dim=0, keepdim=True)
    sd = x.std(dim=0, keepdim=True).clamp_min(eps)
    return (x - mu) / sd


def low_rank_padded_with_noise(n: int, d: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """K independent informative axes, padded with iid noise on remaining D-K."""
    # Random orthonormal basis (Q from a QR decomp)
    A = torch.randn(d, d, generator=gen)
    Q, _ = torch.linalg.qr(A)
    informative_basis = Q[:, :k]               # (D, K)
    noise_basis = Q[:, k:]                     # (D, D-K)
    sig = torch.randn(n, k, generator=gen)     # info content
    nse = torch.randn(n, d - k, generator=gen) # noise content
    return sig @ informative_basis.t() + nse @ noise_basis.t()


def low_rank_padded_with_zero(n: int, d: int, k: int, gen: torch.Generator) -> torch.Tensor:
    """K independent informative axes, the other D-K dims literally zero."""
    A = torch.randn(d, d, generator=gen)
    Q, _ = torch.linalg.qr(A)
    informative_basis = Q[:, :k]
    sig = torch.randn(n, k, generator=gen)
    return sig @ informative_basis.t()


def effective_rank(x: torch.Tensor) -> float:
    xc = x - x.mean(dim=0, keepdim=True)
    sv = torch.linalg.svdvals(xc)
    p = sv / sv.sum().clamp_min(1e-8)
    p = p[p > 1e-12]
    return float(torch.exp(-(p * p.log()).sum()).item())


@torch.no_grad()
def measure(z: torch.Tensor, num_projections: int, num_quadrature: int, repeats: int) -> tuple[float, float]:
    vals = [
        float(sigreg_loss(z, num_projections=num_projections, num_quadrature=num_quadrature).item())
        for _ in range(repeats)
    ]
    t = torch.tensor(vals)
    return float(t.mean()), float(t.std())


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--n", type=int, default=512, help="batch size (samples per dist)")
    p.add_argument("--d", type=int, default=320, help="embedding dim")
    p.add_argument("--projections", type=int, default=1024)
    p.add_argument("--quadrature", type=int, default=32)
    p.add_argument("--repeats", type=int, default=8, help="re-draw projections N times for std")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()

    torch.manual_seed(args.seed)
    gen = torch.Generator().manual_seed(args.seed)

    n, d = args.n, args.d
    print(f"# SIGReg rank diagnostic  (N={n}, D={d}, M={args.projections}, repeats={args.repeats})")
    print(f"#   each row reports: sigreg mean +/- std over {args.repeats} re-draws of the M random directions")
    print()
    header = f"  {'distribution':<46s} {'eff_rank':>9s}  {'sigreg':>20s}"
    print(header)
    print("-" * len(header))

    # 1. Full-rank N(0, I)
    z = torch.randn(n, d, generator=gen)
    mu, sd = measure(z, args.projections, args.quadrature, args.repeats)
    print(f"  {'full-rank N(0, I_D)':<46s} {effective_rank(z):>9.2f}  {mu:>10.5f} +/- {sd:.5f}")

    # 2. Single non-unit dim
    z = torch.randn(n, d, generator=gen)
    z[:, 0] *= 5.0
    mu, sd = measure(z, args.projections, args.quadrature, args.repeats)
    print(f"  {'N(0, I_D) but dim 0 scaled 5x':<46s} {effective_rank(z):>9.2f}  {mu:>10.5f} +/- {sd:.5f}")

    # 3. Rank-K padded with noise + per-dim BN
    print()
    print("  Rank-K signal padded with N(0,1) noise on remaining D-K dims, then per-dim BN'd:")
    for k in [1, 3, 8, 32, 64, 128, 320]:
        z = low_rank_padded_with_noise(n, d, k, gen)
        z = per_dim_bn(z)
        mu, sd = measure(z, args.projections, args.quadrature, args.repeats)
        label = f"    K={k:<3d} (signal in K dirs, noise elsewhere) BN'd"
        print(f"  {label:<46s} {effective_rank(z):>9.2f}  {mu:>10.5f} +/- {sd:.5f}")

    # 4. Rank-K, zero-padded + per-dim BN (BN epsilon turns zero into noise)
    print()
    print("  Rank-K signal, zeros on remaining D-K dims, then per-dim BN'd:")
    for k in [1, 3, 8, 32, 64, 128, 320]:
        z = low_rank_padded_with_zero(n, d, k, gen)
        z = per_dim_bn(z)
        mu, sd = measure(z, args.projections, args.quadrature, args.repeats)
        label = f"    K={k:<3d} (signal in K dirs, zeros elsewhere) BN'd"
        print(f"  {label:<46s} {effective_rank(z):>9.2f}  {mu:>10.5f} +/- {sd:.5f}")

    print()
    print("Interpretation:")
    print(" - Rows 1, K=320 should be the smallest (true unit Gaussian).")
    print(" - If rows for small K (1, 3, 8) padded with noise are NOT")
    print("   substantially larger than full-rank, SIGReg can't see rank")
    print("   collapse when each dim has been BN-normalized to unit variance.")
    print(" - The dim-0-scaled-5x row should spike (single non-unit marginal).")


if __name__ == "__main__":
    main()
