# 1.3B Dion Experiment — GPU Run Guide

This guide explains how to run the 1.3B-scale extension of our 162M experiments. The script is `train_1p3b.py`; it supports gradient accumulation, optional activation checkpointing, and integrated $\kappa(P_t)$ tracking.

## Dion paper 1.3B reference setup (Table 8 of Ahn et al. 2025a)

| Quantity | Value |
|-|-|
| $d_{\mathrm{model}}$ | 2048 |
| Layers | 24 |
| Heads | 32 |
| Total parameters | ~1.4B |
| Batch tokens | 2.1M |
| Total steps | 12,000 |
| Total tokens | 25.2B (Chinchilla-optimal) |
| Learning rate | 0.01 (Dion paper claims 0.01 for all sizes) |

## Dion-managed matrices

With $d=2048$, $r=512$ (rank fraction 0.25), and 24 layers, there are **96 Dion-managed matrices**:

| Layer position | Shape | Count | $P_t$ shape |
|-|-|-|-|
| `attn.qkv` | $(6144, 2048)$ | 24 | $(6144, 512)$ |
| `attn.proj` | $(2048, 2048)$ | 24 | $(2048, 512)$ |
| `mlp.fc` | $(8192, 2048)$ | 24 | $(8192, 512)$ |
| `mlp.proj` | $(2048, 8192)$ | 24 | $(2048, 512)$ |

## Memory and wall-clock estimates

**Memory** on H100 80GB:

| Component | Size |
|-|-|
| Model weights (bf16) | 2.6 GB |
| Gradients (bf16) | 2.6 GB |
| Master copy (fp32) | 5.3 GB |
| Dion state (M, V) | 1.3 GB |
| AdamW state (embed + lm_head) | 1.5 GB |
| **Optimizer subtotal** | **~13 GB** |
| Activations at micro-batch 8, seq 1024 | ~1.5 GB |
| Activations at micro-batch 32, seq 1024 | ~6 GB |
| **Fits comfortably on H100 80GB** ✓ | |

**Wall-clock** on a single H100 (sustained ~400 TFLOP/s bf16):

| Run | Tokens | Wall-clock |
|-|-|-|
| Full Dion-paper 1.3B (12K steps, 25.2B tokens) | 25.2B | **~6.2 days** |
| Half-budget (6K steps, 12.6B tokens) | 12.6B | ~3.1 days |
| **Short comparison (3K steps, 6.3B tokens)** | 6.3B | **~1.6 days** |
| Very short (1K steps, 2.1B tokens) | 2.1B | ~13 hours |

For multi-GPU: divide by number of GPUs (single-node DDP-equivalent). 8× H100 brings the full run down to ~19 hours.

## Recommended runs

### Option A — Full Chinchilla-optimal (Dion paper reproduction)

```bash
python train_1p3b.py \
    --data fineweb \
    --d_model 2048 --n_layers 24 --n_heads 32 \
    --seq_len 1024 --micro_batch 8 --grad_accum 256 \
    --steps 12000 --log_every 100 \
    --lr 0.01 --adam_lr 3e-4 --weight_decay 0.01 \
    --warmup_frac 0.0 --cooldown_frac 0.1 \
    --rank_fraction 0.25 --qr_method qr --qr_warmup_steps 200 \
    --opts dion \
    --out_dir results/dion_1p3b_full
```

This trains a single Dion run for 12K steps × 2.1M tokens = 25.2B tokens. To reproduce Figure 2 of the Dion paper (showing Muon, Dion at three rank fractions), you would run this script three more times with `--rank_fraction 0.5`, `0.0625`, and once with `--opts muon`. Total compute: ~25 GPU-days on a single H100.

### Option B — Short comparison run (most cost-efficient)

```bash
python train_1p3b.py \
    --data fineweb \
    --d_model 2048 --n_layers 24 --n_heads 32 \
    --seq_len 1024 --micro_batch 8 --grad_accum 256 \
    --steps 3000 --log_every 100 \
    --lr 0.01 --adam_lr 3e-4 --weight_decay 0.01 \
    --warmup_frac 0.0 --cooldown_frac 0.1 \
    --rank_fraction 0.25 --qr_method qr --qr_warmup_steps 200 \
    --opts muon dion dion2 \
    --out_dir results/comparison_1p3b_short
```

**3K steps × 2.1M tokens = 6.3B tokens**, three optimizers, ~5 GPU-days total on a single H100. This is the shortest run that reproduces the qualitative pattern of the 162M experiment at 1.3B scale (loss curves, wall-clock comparison).

### Option C — 1.3B with $\kappa(P_t)$ tracking

```bash
python train_1p3b.py \
    --data fineweb \
    --d_model 2048 --n_layers 24 --n_heads 32 \
    --seq_len 1024 --micro_batch 8 --grad_accum 256 \
    --steps 3000 --log_every 100 \
    --lr 0.01 --adam_lr 3e-4 \
    --rank_fraction 0.25 --qr_method qr --qr_warmup_steps 200 \
    --opts dion \
    --track_kappa \
    --kappa_method svd \
    --kappa_stride 100 \
    --kappa_dense_until 200 \
    --out_dir results/dion_1p3b_kappa
```

A single Dion run with $\kappa$ instrumentation. **Configuration choices**:

- **`--kappa_method svd`** (recommended): exact via `torch.linalg.svdvals(P)`. On 1.3B at $r=512$, each SVD is ~1ms on H100; 22K SVDs total ≈ 22 seconds added overhead. The cheaper `power_iter` alternative computes `eigvalsh(P^T P)` whose condition number is $\kappa(P)^2$, and consequently breaks numerically when $\kappa(P) \gtrsim 10^5$ — exactly the pre-warmup transient regime we observed at 162M ($\kappa \approx 6\times 10^5$ at step 2). Use `svd` to avoid losing measurements there.
- **`--kappa_stride 100`** (vs.\ 50 for 162M): the 1.3B model has 96 Dion-managed matrices (vs.\ 48 for 162M); doubling the stride keeps total observations comparable to the 162M experiment (~22K vs.\ ~7.6K for 162M). 
- **`--kappa_dense_until 200`**: aligns with `qr_warmup_steps=200`, captures the pre-warmup transient densely.

The output `kappa_summary.json` will tell you whether the post-warmup $\kappa(P_t)$ stays below $10^3$ at 1.3B scale — the question Section A.1 of the Dion paper explicitly raises and Section 7.4 of our paper lists as an open question for larger models.

### Option D — Very short scaling check (~13 hours)

Same as Option C but with `--steps 1000`. Useful as a first probe to verify everything runs on your hardware before committing days of compute.

## What the kappa-tracking output will show

Compare to the 162M result we already have (Table 6 of our paper):

| Model | Post-warmup max $\kappa$ | Pre-warmup peak |
|-|-|-|
| 162M (measured) | 611 | $6.6 \times 10^5$ |
| 1.3B (this run will measure) | ? | ? |

The Dion paper (Section A.1) reports that $\kappa$ grows with model size; the question is whether at 1.3B it stays below the assumed $\kappa_{\max}=10^3$ or starts to exceed it. Three possible outcomes:

1. **post-warmup max $< 1000$**: Validates Corollary 1 at 1.3B. Strongest possible evidence for our paper's main empirical claim. Update Section 6.5 with a per-model table.
2. **post-warmup max in $[1000, 5000]$**: CQR borderline. The fp32 CQR error bound is approaching the regularity threshold; RCQR (Corollary 2) becomes the safer recommendation. Update Section 6.7 ("Limitations / Model scale below frontier") with the measured value.
3. **post-warmup max $> 5000$**: CQR is unsafe in fp32 at 1.3B; RCQR is necessary. This would be an interesting negative result that cleanly motivates the RCQR analysis. Update paper with a candid discussion.

All three outcomes strengthen the paper. Outcome 1 is most likely based on Dion paper's Section A.1 reading.

## Implementation notes

### Why gradient accumulation?

A 2.1M-token batch at seq_len=1024 = **2050 sequences per step**. No GPU can hold 2050 1.3B-forward activations at once. We accumulate gradients over 256 micro-batches of size 8 (8 × 256 = 2048 ≈ 2050 sequences).

### Why bf16?

The script uses `torch.amp.autocast(dtype=torch.bfloat16)` on CUDA, matching the 162M setup. Master copies stay in fp32 inside Dion's `M` and `V` buffers for numerical stability. The orthonormalization step is the only place where precision matters: at $\kappa\approx 10^3$, both fp32 and fp64 satisfy the regularity condition, as our Section 6.5 analysis shows.

### Why 200-step QR warmup?

Same reason as 162M: the cold-start power-iteration transient produces $\kappa\sim 10^5$--$10^6$ in the first few steps. CQR is unsafe in this regime; QR is unconditionally safe. The default `qr_method=qr` plus `qr_warmup_steps=200` runs plain QR for the first 200 steps and switches to your chosen `qr_method` afterwards. For `qr_method=qr`, the warmup parameter has no effect (QR is used throughout).

### Activation checkpointing

Optional via `--activation_checkpointing`. Reduces activation memory by ~$\sqrt{n_{\mathrm{layers}}}$ at the cost of one extra forward per backward (~30% slower). Not needed at micro-batch 8 on H100 80GB; only enable if you push micro-batch past 32 or run on a smaller GPU.

### Multi-GPU

The script is single-GPU (DDP not added — see Limitations of our paper). For multi-GPU runs, you would wrap the model in `torch.nn.parallel.DistributedDataParallel` and divide `grad_accum` by the world size. The Dion paper's distributed Algorithm~5 (RCQR with $r\times r$ communication) is described in detail there but not implemented here.

## Files saved per run

```
{out_dir}/
├── history.json               # train/val loss + wall-clock per logged step
├── kappa_log.json             # (only if --track_kappa) full per-matrix kappa trajectory  
├── kappa_summary.json         # (only if --track_kappa) aggregate statistics
└── kappa_trajectory.png       # (only if --track_kappa) two-panel plot
```

After the run completes, send back the JSON files and the PNG and we will integrate the 1.3B results into Section 6.5 of the paper, in particular:

- **New row of Table 6**: 162M and 1.3B side by side for post-warmup max kappa
- **Updated paragraph in Section 6.7 Limitations**: replace "open question" with measured value
- **Possibly new figure** if 1.3B trajectory has substantially different shape from 162M
