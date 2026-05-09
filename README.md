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
  datasets.py   make_dataset(name, ...) — real-env trajectory datasets
  eval.py       MPC controller + evaluate(): goal-reaching control eval
  probing.py    Linear/MLP probes from frozen latents to env state
  envs/
    tworoom.py  2D nav with two rooms + doorway (custom)
    reacher.py  2-link planar arm (custom kinematics)
    pusht.py    Push-T via pymunk physics
    cube.py     OGBench cube-single-play-v0 wrapper
train.py        Training entry point with --env flag (--amp for bf16 mixed precision)
eval.py         Control-evaluation entry point (loads checkpoint, runs MPC)
probe.py        Probing entry point (loads checkpoint, fits state regressors)
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
# Pipeline smoke test (synthetic data, < 1s on GPU)
.venv\Scripts\python.exe smoke_test.py

# Train on the synthetic moving-blob dataset (no env deps needed)
.venv\Scripts\python.exe train.py --env synthetic --epochs 5

# Train on a real env (collects trajectories on first run, caches under ./data/)
# --amp turns on bf16 autocast (~1.2x speedup on RTX 3070, more on bigger GPUs)
.venv\Scripts\python.exe train.py --env tworoom      --epochs 30 --num-trajs 1024 --amp
.venv\Scripts\python.exe train.py --env reacher      --epochs 30 --num-trajs 1024 --amp
.venv\Scripts\python.exe train.py --env pusht        --epochs 20 --num-trajs 256  --amp
.venv\Scripts\python.exe train.py --env pusht_expert --epochs 20 --num-trajs 1024 --amp   # downloads ~31MB on first run
.venv\Scripts\python.exe train.py --env cube         --epochs 15 --num-trajs 128  --amp

# Evaluate goal-reaching with MPC + CEM
.venv\Scripts\python.exe eval.py --ckpt checkpoints/lewm_tworoom.pt --episodes 8

# Probe: how well does the frozen encoder's latent decode env state?
.venv\Scripts\python.exe probe.py --ckpt checkpoints/lewm_reacher.pt --frames 4096
```

The smoke test exercises the full training pipeline (forward, backward, AdaLN,
SIGReg, CEM). `eval.py` loads a trained checkpoint and runs receding-horizon
MPC: at each step it plans `--horizon` actions in latent space, executes the
first `--replan-every`, observes the new frame, then replans. Per-episode
output reports state-distance start→end, latent-space distance to goal, and
pixel MSE.

## Environments

| name           | source                              | obs default | action dim | data source                                  |
|----------------|-------------------------------------|-------------|------------|----------------------------------------------|
| `tworoom`      | custom 2D nav (this repo)           | 48 x 48     | 2          | random-policy rollouts collected on first run |
| `reacher`      | custom 2-link arm (this repo)       | 48 x 48     | 2          | random-policy rollouts collected on first run |
| `pusht`        | `pymunk` physics, T-shaped block    | 96 x 96     | 2          | random-policy rollouts collected on first run |
| `pusht_expert` | (data only — uses `pusht` env)      | 96 x 96     | 2          | Diffusion-Policy's released expert dataset (`pusht.zip`, ~31 MB), auto-downloaded |
| `cube`         | `ogbench` `cube-single-play-v0`     | 64 x 64     | 5          | OGBench's released action stream replayed in env, frames re-rendered |
| `synthetic`    | `lewm.data.MovingBlobDataset`       | 56 x 56     | 2          | analytic moving-blob trajectories (no env)    |

Datasets are cached under `./data/{env}_{source}_n{N}_T{T}_S{S}_seed{seed}.pt`.
First run for `cube` will download the ogbench dataset (~270 MB) under
`~/.ogbench/data/`; subsequent runs reuse it.

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

## Probing results (current implementation)

How well does the frozen encoder's latent decode env state? We collect 4096
random-rollout (obs, state) pairs per env, encode, then fit ridge linear and
2-layer MLP regressors to env state on an 80/20 split.

| env            | embed dim | effective rank | linear R² | MLP R² | notes |
|----------------|-----------|----------------|-----------|--------|-------|
| `tworoom`      | 320       | 2.8            | -0.58     | +0.02  | severely collapsed; encoder reduces to ~"which room am I in" |
| `reacher`      | 320       | 4.5            | -30.3     | **+0.55** | tip position recoverable nonlinearly |
| `pusht`        | 320       | 3.4            | -1.63     | +0.22  | random-rollout data is too weak — agent rarely contacts the T |
| `pusht_expert` | 320       | **6.6**        | **+0.44** | **+0.60** | expert demonstrations dramatically improve representation |
| `cube`         | 320       | 4.0            | -0.38     | +0.27  | 24 of 28 state dims (4 are constant) |

Two takeaways:
1. **Effective rank is consistently far below the embed dim.** SIGReg keeps
   each random-projection's marginal Gaussian, but with M=1024 random
   directions on S^319, the rank-3-to-7 informative subspace is sampled
   rarely enough that SIGReg's constraint is satisfied without forcing
   isotropy. Per-dim BN gives unit variance per dim, but the dimensions are
   highly correlated copies of a few underlying signals. This is a real
   finding worth flagging — the regularizer alone doesn't prevent
   dimensional collapse on these envs at our training budgets.
2. **Data quality matters enormously.** Switching from random rollouts to
   real expert demonstrations on Push-T roughly doubles effective rank,
   pushes linear R² from negative to +0.44, and triples MLP R² (0.22→0.60).
   Random actions in contact-rich envs are mostly no-ops; the world model
   has nothing to learn.

A note on BN: the encoder's 1-layer-MLP+BatchNorm projector has stale
running mean/var after training (eval-mode std=4 vs trained std=1). All
three downstream tools fix this:
- `train.py` does a recal pass over the training set as the last step,
  so saved checkpoints have correct running stats.
- `eval.py` does a recal pass on fresh random rollouts at load time
  (configurable via `--bn-recal-frames`, default 1024).
- `probe.py` recalibrates against the probe set itself.

The recal helps eval substantially. On a 30-epoch tworoom checkpoint:

| metric                        | without recal | with recal |
|-------------------------------|---------------|------------|
| MPC success rate (8 ep)       | 0%            | **12.5%**  |
| mean state-distance reduction | -0.06 (hurts) | **+0.02**  |
| mean terminal latent distance | 660           | 196        |

On a 30-epoch reacher checkpoint with recal, MPC achieves a
**+0.35 mean state-distance reduction** (initial 0.76 → terminal 0.41,
i.e. ~46% closer to goal on average) with 12.5% strict success.

## Before increasing the training budget

Our control numbers are well below the paper's. The obvious move is "train
longer" — but throwing compute at a possibly-buggy implementation is
wasteful. Ranked by ROI, the work below should happen first. Tier 1 can
*invalidate* the assumption that more training will help; Tier 2 might
already close part of the gap with no retraining.

### Tier 1 — correctness checks (run; results below)
1. **Verify the model can overfit a single batch.** Take 4 trajectories,
   train 500 steps, expect pred loss ≈ 0. If it can't overfit, no budget
   will help. ~2 min to run. **(`diagnostics/overfit_batch.py`)**
2. **Investigate the rank-3 collapse.** Two competing hypotheses:
   *(a)* SIGReg implementation bug — feed `N(0, I)` and a rank-3
   "isotropic Gaussian padded with noise" through `sigreg_loss` and
   compare. If both look small, the regularizer literally doesn't see
   rank as a problem.
   *(b)* The paper's regularizer is genuinely weak on these envs at
   modest budgets. Either way knowing which is cheap and changes
   everything. **(`diagnostics/sigreg_rank.py`)**
3. **Verify the AdaLN variant.** We implemented per-token AdaLN-Zero
   (each token modulated by its own action). Per-token is the natural
   fit for an autoregressive predictor with one action per token, but
   the paper just says "AdaLN at each layer." Sequence-shared AdaLN
   (DiT-style) is a different inductive bias. ~10 min to write the
   alternative and compare 50-step loss curves on synthetic data.
   **(`diagnostics/adaln_variant.py`)**

### Tier 1 results

**1. Single-batch overfit — PASS.** On a fixed 4-trajectory batch, pred_loss
crashes from 1.96 → 0.0008 in 500 steps (~2400× reduction, 35s on RTX 3070).
Architecture and gradient flow are sound; the model is not bottlenecked by
inability to fit. Eval-mode pred_loss is 0.004 (vs 0.001 train-mode) due to
the BN running-stat lag we've already documented — not a new issue.

**2. SIGReg rank-collapse probe — found the smoking gun.** The regularizer
literally cannot see rank collapse when the encoder pads dead dims with iid
noise. With M=1024 random unit-norm directions on S^319, sigreg readings:

| distribution                                     | eff. rank | sigreg          |
|--------------------------------------------------|-----------|-----------------|
| full-rank N(0, I_320)                            | 288       | **0.00057**     |
| rank-3 signal padded with N(0,1) noise, BN'd     | 289       | **0.00056**     |
| rank-8 signal padded with N(0,1) noise, BN'd     | 288       | **0.00056**     |
| rank-3 signal, zero-padded, BN'd                 | 3.00      | 0.0364 (65× ↑)  |
| rank-1 signal, zero-padded, BN'd                 | 1.00      | 0.0883 (158× ↑) |
| N(0, I_320) but dim 0 scaled 5x                  | 285       | 0.00166 (3× ↑)  |

The rank-3 *padded with noise* row is **indistinguishable from full-rank
Gaussian**, while rank-3 *zero-padded* spikes 65×. The takeaway: SIGReg + per-dim
BN admits a "noise-pad" loophole — the encoder can keep all useful information
in K << D dims and route iid noise through the rest, satisfying every 1D
marginal under random projections while the joint distribution stays low-rank.
Random projections sample the K-dim informative subspace at rate K/D, which is
~1% for K=3, D=320 — too rare for the projected Epps–Pulley statistic to fire.
Our trained encoders' sigreg of ~0.07-0.10 sits between rank-3 zero-pad (0.036)
and rank-1 zero-pad (0.088), suggesting actual encoders use a mix of low-rank
structure plus residual non-Gaussian features (heavy tails, multimodality)
beyond pure rank.

**Why this is structural, not a bug.** The mechanism is geometric. A random
unit vector u ∈ S^(D−1) has E[u_i²] = 1/D, so a projection u·z splits energy
as ≈ K/D onto a K-dim informative subspace and (D−K)/D onto everything else.
With BN forcing each dim to unit variance, the (D−K) noise-pad coordinates
sum to a CLT-Gaussian-looking quantity that dominates the projection.
Non-Gaussianity in the K signal dims is suppressed in the projection by
~√(K/D) ≈ 0.1 for K=3, D=320; higher-moment detectors suppress it further.
Each individual component of the loss is doing what it's specified to do
(BN: per-dim moment match; SIGReg: 1D-marginal Gaussianity). The loophole
lives in what they jointly *don't* constrain: joint covariance. The paper's
two-term loss is underspecified for representations whose downstream use is
L2-distance-based (CEM cost = ‖z − z_g‖² across all D dims), not wrong.

**Why this matters for control.** CEM's planning cost is `‖z_pred(actions) −
z_goal‖²` summed across all D=320 dims. With K useful dims and D−K noise-pad
dims, signal-to-noise in the cost is ~K/D ≈ 1% — CEM is searching action
sequences that minimize a function whose ~99% of magnitude is junk. The K
signal dims also need to encode state in a way an L2 metric can use; our
probing shows they don't (tworoom: linear R² = −0.58, MLP R² = +0.02 — the
information is there but tangled). So both rank *and* geometry within the
used subspace contribute to the gap.

**What scale does and does not change.** Batch size N drops the empirical
char-fn noise floor as ~1/N; below that floor SIGReg can't separate
"true Gaussian" from "Gaussian-looking with √(K/D)-suppressed
contamination." Larger N tightens the residual, but the suppression
factor is geometric and independent of N — N helps marginally, not
structurally. M (number of projections) reduces variance of the mean stat
but not its bias: the mean is dominated by the ~99% of directions in the
noise subspace whether M=1024 or 10000. Longer training does not close
the loophole — the encoder is not pushed away from the noise-pad mode by
SGD. What more training *can* do is reduce absolute pred_loss and
incidentally raise rank if env intrinsic state dim and data are rich
enough that prediction needs it. For our toy envs (tworoom 2D, reacher
4D) prediction does not need high rank; the encoder correctly identifies
the state is low-dim.

**Closing it within paper spec, by leverage:**
1. **Shrink pred_dim toward env intrinsic state dim** (biggest in-spec
   lever). The contamination factor is √(K/D); cutting D from 320 → 32
   for low-dim envs lifts it from 0.1 to 0.3 — projections start landing
   close enough to the signal subspace for non-Gaussianity to show up.
   The paper specifies the loss formula and architecture, not that
   pred_dim must be 320 for every env.
2. **Raise λ** (paper's stated knob). Doesn't make SIGReg see new
   structure but tightens residual error on what it already sees (tail
   mass, multimodality). Sweep {0.1, 1.0, 10.0}.
3. **Larger batch** (e.g. 128). Drops noise floor; free with AMP.

**Closing it deviating from paper.** Add `‖(1/N)ZᵀZ − I_D‖²` to the
loss. SIGReg constrains 1D marginals; this constrains the joint second
moment. Together they pin both rank and per-dim variance. But this
makes it a three-term loss, breaking the paper's headline claim. Worth
reaching for only if all three in-spec levers fail.

**3. AdaLN per-token vs sequence-shared — per-token wins.** 50 steps on
synthetic, identical seeds, batch 32, T=8:

| step | per-token pred | shared pred |
|------|----------------|-------------|
| 0    | 2.02           | 1.99        |
| 14   | 0.17           | 0.16        |
| 29   | 0.11           | 0.12        |
| 49   | **0.047**      | 0.114       |

Both crash similarly through step ~20 (the easy "average dynamics" regime
where per-step actions don't matter much), then per-token pulls away by
~2.4× at step 50. Per-token is the right inductive bias for autoregressive
next-step prediction with per-step actions. Current implementation is correct.

**Net implication**: the gap to the paper is not a bug in overfitting
capacity, predictor architecture, or SIGReg correctness. It is the
SIGReg+BN noise-pad loophole, which is structural and unaffected by
training scale. The recommended order is now: Tier 2 (eval-side sweeps,
no retraining) and Tier 3 #6/#7/#10 (the three in-spec levers — λ, batch,
shrink pred_dim) in parallel. The covariance term is a last resort that
costs the paper's two-term claim.

### Tier 2 — eval-side wins (no retraining needed)
4. **Sweep CEM hyperparameters at eval time.** Currently the smoke
   tests use 128 samples × 8 elites × 8 iters. Paper-spec is
   300 × 30 × 30. Try the full spec and sweep `--horizon` 4/8/12/16
   and `--replan-every` 1/2/4. Each point ~100s. The reacher result
   (+0.35 mean state-distance reduction) is already promising — tuned
   CEM might push it past 50% strict success without touching training.
5. **Sweep `--success-threshold`.** Our defaults (0.08 for
   tworoom/reacher) are arbitrary tight. Reporting curves of
   success-rate vs threshold is more honest and lets us compare to
   whatever number the paper actually uses.

### Tier 3 — in-spec attacks on the noise-pad loophole
6. **Sweep SIGReg λ** (0.1, 1.0, 10.0). Paper's stated tunable knob.
   Doesn't make SIGReg see new structure but tightens the residual on
   what it does see (tail mass, multimodality). May force the encoder
   to spread non-Gaussian features across more dims.
7. **Increase batch size** (128 or 256). Drops the empirical char-fn
   noise floor (~1/N), making subtler deviations visible above the
   Monte-Carlo noise. Free with AMP.
8. **Log effective rank during training.** Print every N steps. Lets
   us see whether rank ever rises during training or is locked by
   step ~20, which determines whether other levers are even biting.
9. **Sanity-check that prediction loss continues to fall** alongside
   any rank changes. The loophole means rank can rise while pred_loss
   doesn't — but if pred_loss stops falling when we tighten λ, we've
   gone too far.
10. **Shrink `pred_dim` toward env intrinsic state dim** (highest
    leverage on the loophole). Contamination factor in random
    projections is √(K/D); cutting D from 320 → 32 for low-state-dim
    envs lifts it from 0.1 to ~0.3 — projections now land close enough
    to the signal subspace for non-Gaussianity to register.
    Suggested per-env: tworoom 32, reacher 64, pusht 128, cube 192.

### Tier 4 — only after the above
11. Increase training budget toward the paper's "hours" (~30 min/env).
    Note that scale alone does not close the loophole; it can only
    raise rank incidentally if prediction needs it.

### Tier 5 — paper-deviating, last resort
12. **Add covariance term** `‖(1/N)ZᵀZ − I_D‖²` to the loss. Constrains
    joint second moment; together with SIGReg pins both rank and
    per-dim variance. Closes the loophole at the source but costs the
    paper's "two-term loss" headline claim.

## Open todos

What's done so far is the **training pipeline** — encoder, predictor with
AdaLN-Zero, SIGReg, the four envs, datasets, the CEM planner, and a
end-to-end smoke test. What's still missing relative to the paper:

### Evaluation / control
- [x] **Control evaluation loop with MPC.** `lewm/eval.py` + `eval.py` CLI
  reset the env, encode a goal observation, run receding-horizon MPC over CEM,
  and report success rate / state-distance reduction / terminal latent
  distance. Tuning of CEM / horizon hyperparameters per env is still open.
- [x] **Per-env success criteria.** Each env exposes `get_state()` and
  `state_distance(goal_state)` (tworoom: 2D position; reacher: tip xy; pusht:
  block (x,y,θ); cube: 28-dim ogbench state).
- [x] **Probing experiments.** `lewm/probing.py` + `probe.py` CLI: collects
  (obs, state) pairs, encodes with frozen encoder (after a BN re-cal pass),
  fits ridge linear + 2-layer MLP regressors, reports per-dim R² and the
  embedding's effective rank. See "Probing results" above.
- [ ] **Violation-of-expectation surprise detection.** The paper detects
  physically implausible events via the predictor's residual. Not implemented.

### Data faithfulness
- [x] **Real Push-T expert data.** `--env pusht_expert` auto-downloads
  Diffusion-Policy's `pusht.zip` (~31 MB), normalizes the action range to
  [-1, 1], and slides 16-step windows. Probing shows it's substantially
  better than the random-rollout `pusht`.
- [ ] **Push-T action-modality mismatch**: the expert dataset's actions
  are *target end-effector positions* (DP's control mode), but our
  `lewm/envs/pusht.py` treats actions as *velocity commands*. So a model
  trained on `pusht_expert` and evaluated against our `pusht` env doesn't
  steer correctly — CEM finds plans that look fine in latent space but the
  env interprets them differently. To close this gap, switch our pusht env
  to position-command control (matching DP's reference) or convert the
  expert action stream into velocities before normalization.
- [ ] Custom Two-Room and Reacher envs are minimal-but-faithful shapes /
  dynamics and won't byte-match any specific reference implementation. (No
  public LeWM-released datasets are known for these.)

### Engineering / training quality-of-life
- [x] **Mixed precision (AMP).** `train.py --amp` enables bf16 autocast.
  Modest 1.2× speedup on the RTX 3070 (small model, Tensor Cores
  underutilized); larger gains expected on bigger GPUs / models.
- [ ] **wandb / TensorBoard logging.** Currently just `print` to stdout.
- [ ] **Multi-GPU / DDP.** Single-GPU only.
- [ ] **KV caching in `LeWorldModel.rollout_latent`.** Each CEM step re-runs
  the full predictor (O(t²) for t-step rollouts). Fine for short horizons,
  slow for long ones. KV caching would make CEM with H>>10 tractable.
- [ ] **Streaming / num_workers > 0 data pipeline.** Datasets currently sit
  in CPU memory as uint8; fine at our sizes, would need work for larger.

### Hyperparameters
- The paper abstract doesn't pin down LR, batch size, or exact image
  resolutions per env. We use AdamW, lr 3e-4, wd 0.05 and the resolutions
  shown in the env table. `train.py` auto-adjusts `patch_size` if `img_size`
  isn't divisible by 14.

## Reference

```
@article{maes2026lewm,
  title  = {LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels},
  author = {Maes, Lucas and Le Lidec, Quentin and Scieur, Damien and LeCun, Yann and Balestriero, Randall},
  journal= {arXiv preprint arXiv:2603.19312},
  year   = {2026}
}
```
