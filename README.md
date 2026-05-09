# LeWorldModel

A from-scratch PyTorch implementation of **"LeWorldModel: Stable End-to-End
Joint-Embedding Predictive Architecture from Pixels"**
([arXiv:2603.19312](https://arxiv.org/abs/2603.19312), Maes, Le Lidec, Scieur,
LeCun, Balestriero).

LeWM is a Joint Embedding Predictive Architecture (JEPA) that learns a world
model directly from raw pixels with only **two loss terms** — a next-embedding
prediction loss and an isotropic-Gaussian regularizer (SIGReg) — and avoids
representation collapse without EMAs, stop-gradients, or pre-trained encoders.

## Method, in one screen

```
obs_t  ─ encoder(ViT-tiny + BN-projector) ─►  z_t  ∈ ℝ^D
                                              │
                  a_t ──► AdaLN-Zero ──► predictor(causal Transformer) ─► ẑ_{t+1}

L = ‖ẑ_{t+1} − z_{t+1}‖²  +  λ · SIGReg(z)
```

**SIGReg** projects embeddings onto M random unit directions and tests each 1D
marginal against N(0, 1) using the **Epps–Pulley** statistic (Gaussian-weighted
L² distance between empirical and standard-normal characteristic functions,
evaluated by trapezoid quadrature). By Cramér–Wold, agreement of every 1D
marginal implies agreement of the joint distribution — so SIGReg pulls the
embedding distribution toward isotropic standard Gaussian and prevents
collapse without any auxiliary networks.

**Planning** is Cross-Entropy Method (CEM) over horizon-`H` action sequences,
rolled out *in latent space* using the predictor; cost is the squared distance
to the goal embedding `z_g = encoder(o_g)`.

## Repo layout

```
lewm/
  encoder.py    ViT-tiny + Linear/BatchNorm projector
  predictor.py  Causal Transformer with per-token AdaLN-Zero action conditioning
  sigreg.py     Epps-Pulley regularizer with random 1D projections
  model.py      LeWorldModel (forward + two-term loss) and LeWMConfig
  planner.py    CEM planner over latent rollouts
  data.py       MovingBlobDataset — synthetic toy trajectories for smoke tests
train.py        Training entry point
smoke_test.py   End-to-end pipeline check (a few training steps + one CEM call)
requirements.txt
```

## Setup

Tested on Windows 11 + Python 3.12 + RTX 3070 (CUDA 12.4 build of PyTorch).

```powershell
python -m venv .venv
.venv\Scripts\python.exe -m pip install --upgrade pip
.venv\Scripts\python.exe -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
.venv\Scripts\python.exe -m pip install -r requirements.txt
```

For CPU-only, drop the `--index-url` flag (PyPI default is CPU build).

## Run

```powershell
.venv\Scripts\python.exe smoke_test.py
.venv\Scripts\python.exe train.py --epochs 5 --batch-size 32
```

Smoke test exercises the full pipeline (forward, backward, AdaLN, SIGReg, CEM)
on a tiny config in under a second on GPU.

## Default config

`LeWMConfig` defaults to roughly the paper's sizes:

| component  | shape                                  | params  |
|------------|----------------------------------------|---------|
| encoder    | ViT-tiny: 12L × 3H × 192d, patch 14    | ~5.5M   |
| predictor  | 6L × 16H × 320d, AdaLN-Zero, 10% drop  | ~11.2M  |
| **total**  |                                        | **~17M** |

(Paper reports ~5M / ~10M / ~15M.) `pred_dim`, `pred_depth`, `pred_heads` are
all configurable via `LeWMConfig`. SIGReg uses M=1024 random projections, 32
quadrature nodes on [0.2, 4], bandwidth 1, λ = 0.1.

## Status — what's stubbed

- The four paper environments (Push-T, OGBench-Cube, Two-Room, Reacher) are
  **not** wired up. `lewm/data.py` ships a synthetic `MovingBlobDataset`
  sufficient to verify the pipeline trains and CEM plans coherent action
  sequences. To use a real env, write a `Dataset` that returns
  `(obs: (T, C, H, W), actions: (T-1, A))` and swap it into `train.py`.
- Some hyperparameters not specified in the paper abstract (LR, batch size,
  exact resolutions per env) use reasonable defaults: AdamW, lr 3e-4, wd 0.05,
  `img_size=56` (must be a multiple of `patch_size=14`).
- Image-space probing / violation-of-expectation evaluation from the paper
  is not implemented.

## Reference

```
@article{maes2026lewm,
  title  = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal= {arXiv preprint arXiv:2603.19312},
  year   = {2026}
}
```
