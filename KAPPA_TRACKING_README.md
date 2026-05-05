# Kappa(P_t) Tracking on the 162M GPT Model

This script directly tests the assumption $\sup_t \kappa(P_t) \le \kappa_{\max}$ that underlies our paper's Corollary 1 (Dion-CQR concrete bound).

## What it does

Runs the same 162M Dion training as `train_compare.py --opts dion`, but additionally records $\kappa(P_t) = \sigma_{\max}(P_t)/\sigma_{\min}(P_t)$ at every Dion step for every Dion-managed weight matrix. Outputs a JSON log, summary statistics, and a trajectory plot.

The kappa values **do not** affect the Dion update — they are only logged. Training behaviour is identical to `train_compare.py` modulo a small wall-clock overhead for the SVDs.

## Usage

### Recommended setting (matches our 162M experiment)

```bash
python kappa_tracking_162m.py \
    --data fineweb \
    --d_model 768 --n_layers 12 --n_heads 12 \
    --batch_size 8 --seq_len 1024 \
    --steps 3000 --log_every 100 \
    --lr 0.01 --adam_lr 3e-4 \
    --rank_fraction 0.25 \
    --qr_method qr --qr_warmup_steps 0 \
    --kappa_stride 50 \
    --kappa_dense_until 100 \
    --out_dir results/kappa_162m
```

This reproduces the exact 162M / FineWeb-Edu setup of `train_compare.py` (the one that produced `all_methods_sbs.png`), with kappa logging added.

### What the kappa-specific arguments do

| Argument | Default | Effect |
|-|-|-|
| `--kappa_stride` | 50 | Record $\kappa$ every this many steps (after the dense window). Stride 50 over 3000 steps = ~60 measurements per matrix, sufficient for the trajectory plot. |
| `--kappa_dense_until` | 100 | Record $\kappa$ at **every** step until this step number, to capture the power-iteration warmup transient. |
| `--kappa_method` | `svd` | Either full SVD (`svd`, exact) or eigenvalues of $P^\top P$ (`power_iter`, slightly cheaper). |

### Cost estimates

For 162M Dion (~50 Dion-managed matrices, mostly $(3072, 192)$, $(768, 192)$):
- Each SVD: ~0.5 ms on H100
- Per Dion step (with logging on): ~50 SVDs ≈ 25 ms additional
- Dense window (steps 1-100): 100 × 25 ms = 2.5 s
- Strided window (steps 101-3000, every 50): ~58 × 25 ms = 1.5 s
- **Total overhead: ~4 s on top of the ~588 s Dion-QR run** (negligible)

### Cheaper variant if needed

```bash
python kappa_tracking_162m.py \
    [...same as above...] \
    --kappa_stride 100 \
    --kappa_dense_until 50 \
    --kappa_method power_iter
```

The `power_iter` method computes $\kappa$ from `eigvalsh(P^T P)` instead of `svdvals(P)`, ~2× faster.

### Smoke test (no GPU needed)

```bash
python kappa_tracking_162m.py \
    --data synthetic --steps 50 \
    --d_model 128 --n_layers 2 --n_heads 4 \
    --batch_size 4 --seq_len 64 --vocab_size 256 \
    --kappa_stride 10 --kappa_dense_until 20 \
    --out_dir results/kappa_smoke
```

Should complete in ~5 seconds on CPU and produce the same output structure.

## Outputs

```
{out_dir}/
├── kappa_log.json        # per-matrix, per-step kappa values
├── kappa_summary.json    # aggregate statistics (max, p99, p95, ...)
├── kappa_trajectory.png  # plot, log y-axis, with reference lines at κ=200, κ=1000
└── history.json          # standard training loss curves (val_loss, wall_time)
```

The summary JSON has the form:

```json
{
  "overall": {
    "n_params": 48,
    "n_total_observations": 2880,
    "max": 873.4,
    "median": 32.1,
    "p95": 412.5,
    "p99": 678.9,
    "post_warmup_max": 873.4,
    "post_warmup_median": 32.1,
    "post_warmup_p95": 412.5,
    "post_warmup_p99": 678.9
  },
  "per_param": {
    "blocks.5.mlp.fc.weight": { "max": 873.4, "median": 245.0, ... },
    ...
  }
}
```

## What the result tells us

| Outcome | Interpretation |
|-|-|
| post-warmup max < 1000 | Validates Corollary 1's $\kappa_{\max} = 10^3$ assumption directly on 162M. |
| post-warmup max in [200, 1000] | Reproduces Ahn et al. 2025a Sec. A.1 |
| post-warmup max >> 1000 | Would require revising the assumption; possibly use Corollary 2 (RCQR) instead, which has no $\kappa$ dependence. |

We **expect** the first two, based on smoke tests on smaller models and Ahn et al.'s reported numbers.

## After the run

Update Section 6.5 of the paper with:
- The post-warmup max from `kappa_summary.json`
- The figure `kappa_trajectory.png`
- The number of Dion-managed matrices (`n_params`) and total observations (`n_total_observations`)

This resolves the "Empirical $\kappa(P_t)$ measured at small scale" caveat in Section 6.7.
