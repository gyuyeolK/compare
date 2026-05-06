"""
1.3B-scale training driver, with gradient accumulation, activation
checkpointing, and optional kappa(P_t) tracking.

Designed to extend the 162M training of train_compare.py to the 1.3B
Dion paper setup (Table 8 of Ahn et al. 2025a):

    Model: d_model=2048, 24 layers, 32 heads
    Batch size: 2.1M tokens   (achieved via gradient accumulation)
    Default total: 12K steps, 25.2B tokens (Chinchilla-optimal)

This is *days* of work on a single H100 if you run the full Dion-paper
schedule. The script accepts arbitrary --steps and --tokens_per_step for
shorter runs; see RUN_GUIDE.md.

The kappa(P_t) tracking is integrated as an optional flag (--track_kappa).
When enabled, it monkey-patches optimizers._safe_orthonormalize, exactly
as in kappa_tracking_162m.py, but with safer defaults (sparser stride,
power_iter method) since the 1.3B model has 96 Dion-managed matrices.

Usage examples
--------------
# Quick smoke test on synthetic (no GPU needed):
python train_1p3b.py --data synthetic --steps 5 --d_model 256 --n_layers 4 \\
    --micro_batch 2 --grad_accum 2 --opts dion

# Full Dion-paper 1.3B run on 1xH100 (~6 days):
python train_1p3b.py --data fineweb --steps 12000 \\
    --d_model 2048 --n_layers 24 --n_heads 32 \\
    --seq_len 1024 --micro_batch 8 --grad_accum 256 \\
    --lr 0.01 --warmup_frac 0.0 --cooldown_frac 0.1 \\
    --rank_fraction 0.25 --qr_method qr --qr_warmup_steps 200 \\
    --opts dion --activation_checkpointing \\
    --out_dir results/dion_1p3b

# Shorter comparison run (3K steps, ~6.3B tokens, ~1.5 days on 1xH100):
python train_1p3b.py --data fineweb --steps 3000 [...same...] \\
    --opts muon dion dion2 --out_dir results/comparison_1p3b_short

# 1.3B with kappa tracking (one Dion run, ~1.6 days on 1xH100, 3000 steps):
python train_1p3b.py --data fineweb --steps 3000 [...same...] \\
    --opts dion --track_kappa --kappa_stride 100 --kappa_dense_until 200 \\
    --kappa_method power_iter --out_dir results/dion_1p3b_kappa
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch
from torch.utils.checkpoint import checkpoint as ckpt_fn

import optimizers   # original optimizers.py, untouched
from optimizers import Muon, Dion, Dion2, split_params
from model import GPT, GPTConfig
from data import get_loader
from train_compare import (
    make_config, build_optimizers, lr_schedule, evaluate as _evaluate_unused
)


# =============================================================================
# Activation checkpointing (optional, for very tight memory)
# =============================================================================

def install_activation_checkpointing(model):
    """Wrap each transformer Block.forward in torch.utils.checkpoint.

    Reduces activation memory roughly by sqrt(n_layers) at the cost of
    one extra forward per backward. Only needed at extreme micro-batch
    sizes; at micro_batch=8, seq_len=1024, 1.3B fits on H100 80GB
    without checkpointing.
    """
    from model import Block  # the transformer block class
    for blk in model.blocks:
        original_forward = blk.forward
        def make_ckpt_forward(orig):
            def ckpt_forward(x, cos, sin):
                # use_reentrant=False is the modern API
                return ckpt_fn(orig, x, cos, sin, use_reentrant=False)
            return ckpt_forward
        blk.forward = make_ckpt_forward(original_forward)
    return model


# =============================================================================
# Kappa(P_t) logger (integrated, identical interface to kappa_tracking_162m.py)
# =============================================================================

class KappaLogger:
    def __init__(self, kappa_method="svd", record_first_k=None):
        self.log = {}
        self.current_step = 0
        self.recording = True
        self.kappa_method = kappa_method
        self._call_order_in_step = 0
        self.record_first_k = record_first_k
        self.shape_key_to_name = {}

    def reset_step(self, step):
        self.current_step = step
        self._call_order_in_step = 0

    @torch.no_grad()
    def measure_kappa(self, P):
        try:
            if self.kappa_method == "svd":
                svals = torch.linalg.svdvals(P.float())
                s_max = float(svals[0])
                s_min = float(svals[-1])
            elif self.kappa_method == "power_iter":
                # cheaper: eigvalsh of P^T P (or P P^T, whichever is smaller)
                m, r = P.shape
                if r <= m:
                    G = P.float().T @ P.float()  # (r, r)
                else:
                    G = P.float() @ P.float().T  # (m, m)
                e = torch.linalg.eigvalsh(G)
                s_max = float(e[-1].clamp_min(0).sqrt())
                s_min = float(e[0].clamp_min(0).sqrt())
            else:
                raise ValueError(f"unknown kappa_method: {self.kappa_method}")
            if s_min > 1e-30:
                return s_max / s_min
            return float("inf")
        except Exception:
            return float("nan")

    def record(self, P):
        if not self.recording:
            self._call_order_in_step += 1
            return
        shape_key = f"shape{tuple(P.shape)}_pos{self._call_order_in_step}"
        if (self.record_first_k is not None
                and len(self.log) >= self.record_first_k
                and shape_key not in self.log):
            self._call_order_in_step += 1
            return
        kappa = self.measure_kappa(P)
        self.log.setdefault(shape_key, []).append({
            "step": self.current_step,
            "kappa": kappa,
            "shape": list(P.shape),
        })
        self._call_order_in_step += 1

    def to_json_safe(self):
        out = {}
        for k, traj in self.log.items():
            cleaned = []
            for entry in traj:
                v = entry["kappa"]
                if isinstance(v, float) and math.isnan(v):
                    v = None
                elif isinstance(v, float) and math.isinf(v):
                    v = "inf"
                cleaned.append({"step": entry["step"], "kappa": v, "shape": entry["shape"]})
            out[k] = cleaned
        return out


_kappa_logger_global = None
_original_safe_orthonormalize = optimizers._safe_orthonormalize


def _patched_safe_orthonormalize(P, method):
    if _kappa_logger_global is not None:
        _kappa_logger_global.record(P)
    return _original_safe_orthonormalize(P, method)


def build_shape_key_to_name_map(model, dion_param_names, rank_fraction=0.25):
    mapping = {}
    for pos, (name, param) in enumerate(dion_param_names):
        m, n = param.shape
        r = max(1, int(round(rank_fraction * min(m, n))))
        shape_key = f"shape{(m, r)}_pos{pos}"
        mapping[shape_key] = name
    return mapping


# =============================================================================
# Evaluation (re-implemented; train_compare.evaluate uses fineweb_loader's
# generator, which is single-pass; we use a fresh val_iter on each call)
# =============================================================================

@torch.no_grad()
def evaluate(model, val_loader_fn, device, num_batches=10):
    model.eval()
    losses = []
    val_iter = val_loader_fn()
    for _ in range(num_batches):
        try:
            xb, yb = next(val_iter)
        except StopIteration:
            break
        with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda")):
            _, loss = model(xb, yb)
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


# =============================================================================
# Train one optimizer with gradient accumulation
# =============================================================================

def train_one(opt_name, args, device, kappa_logger=None):
    """Train with gradient accumulation. One 'step' = grad_accum micro-batches.

    Effective batch tokens = micro_batch * seq_len * grad_accum.
    """
    print(f"\n========== Training with {opt_name.upper()} ==========")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    cfg = make_config(args)
    model = GPT(cfg).to(device)
    if args.activation_checkpointing:
        install_activation_checkpointing(model)
        print("Installed activation checkpointing")
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model: {n_param/1e6:.1f}M parameters "
          f"(d_model={cfg.d_model}, layers={cfg.n_layers}, heads={cfg.n_heads})")
    print(f"Effective batch: micro={args.micro_batch} × grad_accum={args.grad_accum} "
          f"= {args.micro_batch * args.grad_accum} sequences "
          f"= {args.micro_batch * args.grad_accum * args.seq_len:,} tokens/step")

    # Identify Dion-managed parameters in optimizer iteration order
    matrix_params, _ = split_params(model)
    matrix_param_set = set(id(p) for p in matrix_params)
    dion_param_names = [(name, p) for name, p in model.named_parameters()
                        if id(p) in matrix_param_set]
    print(f"Dion-managed matrices: {len(dion_param_names)}")
    if kappa_logger is not None:
        kappa_logger.shape_key_to_name = build_shape_key_to_name_map(
            model, dion_param_names, rank_fraction=args.rank_fraction)

    m_opt, s_opt = build_optimizers(model, opt_name, args)

    def make_train_iter():
        return get_loader(args.data, args.micro_batch, args.seq_len,
                          cfg.vocab_size, device, seed=args.seed)
    def make_val_iter():
        return get_loader(args.data, args.micro_batch, args.seq_len,
                          cfg.vocab_size, device, seed=args.seed + 1)

    train_iter = make_train_iter()
    history = {"step": [], "train_loss": [], "val_loss": [], "wall_time": []}
    t_start = time.time()
    base_m_lr = args.lr
    base_s_lr = args.adam_lr

    model.train()
    for step in range(args.steps):
        # learning-rate schedule
        sched = lr_schedule(step, args.steps,
                            warmup_frac=args.warmup_frac,
                            cooldown_frac=args.cooldown_frac)
        for pg in m_opt.param_groups:
            pg["lr"] = base_m_lr * sched
        for pg in s_opt.param_groups:
            pg["lr"] = base_s_lr * sched

        # === Gradient accumulation ===
        m_opt.zero_grad(set_to_none=True)
        s_opt.zero_grad(set_to_none=True)
        accum_loss = 0.0
        for accum_idx in range(args.grad_accum):
            xb, yb = next(train_iter)
            with torch.amp.autocast(
                    device_type="cuda" if device.type == "cuda" else "cpu",
                    dtype=torch.bfloat16,
                    enabled=(device.type == "cuda")):
                _, loss = model(xb, yb)
                loss = loss / args.grad_accum
            loss.backward()
            accum_loss += loss.item()

        # gradient clipping for AdamW group only
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in s_opt.param_groups for p in g["params"]],
                args.grad_clip,
            )

        # === Tell kappa logger whether to measure on this step ===
        if kappa_logger is not None:
            kappa_logger.reset_step(step + 1)
            kappa_logger.recording = (
                (step + 1) % args.kappa_stride == 0
                or step + 1 <= args.kappa_dense_until
            )

        m_opt.step()
        s_opt.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            wall = time.time() - t_start
            val = evaluate(model, make_val_iter, device,
                           num_batches=args.eval_batches)
            history["step"].append(step + 1)
            history["train_loss"].append(accum_loss)  # already averaged
            history["val_loss"].append(val)
            history["wall_time"].append(wall)
            etas = wall / (step + 1) * (args.steps - step - 1)
            kappa_obs = (sum(len(t) for t in kappa_logger.log.values())
                         if kappa_logger else 0)
            extra = f" | kappa_obs={kappa_obs}" if kappa_logger else ""
            print(f"  step {step+1:5d}/{args.steps} | "
                  f"train {accum_loss:.4f} | val {val:.4f} | "
                  f"lr_mult {sched:.3f} | {wall:.0f}s | "
                  f"ETA {etas/3600:.1f}h{extra}")

    return history


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data", choices=["synthetic", "fineweb"],
                        default="fineweb")
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--micro_batch", type=int, default=8,
                        help="micro-batch size (sequences per forward). "
                             "Total tokens-per-step = micro_batch * seq_len * grad_accum.")
    parser.add_argument("--grad_accum", type=int, default=256,
                        help="gradient-accumulation steps. Default 256 + "
                             "micro=8 + seq=1024 = 2.1M tokens/step (Dion 1.3B).")
    # model: Dion-paper 1.3B defaults
    parser.add_argument("--d_model", type=int, default=2048)
    parser.add_argument("--n_layers", type=int, default=24)
    parser.add_argument("--n_heads", type=int, default=32)
    parser.add_argument("--mlp_mult", type=int, default=4)
    # training: Dion-paper 1.3B defaults
    parser.add_argument("--steps", type=int, default=12000,
                        help="number of optimizer steps (Dion 1.3B paper: 12K).")
    parser.add_argument("--warmup_frac", type=float, default=0.0)
    parser.add_argument("--cooldown_frac", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=20)
    parser.add_argument("--activation_checkpointing", action="store_true",
                        help="enable for very tight memory; slower but reduces activation footprint")
    # optimizers
    parser.add_argument("--opts", nargs="+",
                        default=["dion"],
                        choices=["muon", "dion", "dion2"])
    parser.add_argument("--lr", type=float, default=0.01,
                        help="LR for matrix optimizer (Dion paper: 0.01 for all sizes)")
    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--rank_fraction", type=float, default=0.25)
    parser.add_argument("--qr_method", choices=["qr", "cholesky", "rcqr"],
                        default="qr")
    parser.add_argument("--qr_warmup_steps", type=int, default=200)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--selection", choices=["l1", "random"], default="l1")
    # kappa tracking
    parser.add_argument("--track_kappa", action="store_true",
                        help="enable kappa(P_t) instrumentation. Adds SVD overhead.")
    parser.add_argument("--kappa_stride", type=int, default=100,
                        help="record kappa every this many steps (after dense window). "
                             "Default 100; at 1.3B, 96 matrices × 120 timestamps = 11K obs.")
    parser.add_argument("--kappa_dense_until", type=int, default=200,
                        help="dense kappa recording until step this. Default 200 "
                             "(matches qr_warmup_steps default).")
    parser.add_argument("--kappa_method", choices=["svd", "power_iter"],
                        default="svd",
                        help="default 'svd' for accuracy. 'power_iter' is "
                             "~2x faster but breaks at kappa > ~1e5 (the "
                             "pre-warmup transient regime); only use it if "
                             "the kappa overhead is a real bottleneck.")
    parser.add_argument("--kappa_record_first_k", type=int, default=None)
    # output
    parser.add_argument("--out_dir", default="results/run_1p3b")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory / 1024**3:.0f} GB)")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # ===== Estimate cost =====
    tokens_per_step = args.micro_batch * args.seq_len * args.grad_accum
    total_tokens = tokens_per_step * args.steps
    print(f"\nConfig: {args.steps} steps × {tokens_per_step:,} tokens/step "
          f"= {total_tokens/1e9:.1f}B total tokens")
    if args.track_kappa:
        print(f"kappa tracking: method={args.kappa_method}, stride={args.kappa_stride}, "
              f"dense_until={args.kappa_dense_until}")

    # ===== Optionally install kappa-tracking patch =====
    global _kappa_logger_global
    kappa_logger = None
    if args.track_kappa:
        if "dion" not in args.opts:
            print("WARNING: --track_kappa is meaningful only for 'dion' (not muon/dion2). "
                  "Patching anyway; logger will be empty for non-Dion optimizers.")
        kappa_logger = KappaLogger(
            kappa_method=args.kappa_method,
            record_first_k=args.kappa_record_first_k,
        )
        _kappa_logger_global = kappa_logger
        optimizers._safe_orthonormalize = _patched_safe_orthonormalize
        print(f"Installed kappa-logging patch")

    # ===== Run all requested optimizers =====
    all_history = {}
    for opt in args.opts:
        all_history[opt] = train_one(opt, args, device,
                                     kappa_logger=(kappa_logger if opt == "dion" else None))

    # ===== Save training history =====
    history_path = Path(args.out_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump({"args": vars(args), "history": all_history}, f, indent=2)
    print(f"\nSaved training history to {history_path}")

    # ===== Save kappa log + summary =====
    if kappa_logger is not None and len(kappa_logger.log) > 0:
        log_path = Path(args.out_dir) / "kappa_log.json"
        with open(log_path, "w") as f:
            json.dump({
                "shape_key_to_name": kappa_logger.shape_key_to_name,
                "log": kappa_logger.to_json_safe(),
                "args": vars(args),
            }, f, indent=2)
        print(f"Saved kappa log to {log_path}")

        summary = compute_kappa_summary(
            kappa_logger.log, kappa_logger.shape_key_to_name,
            kappa_dense_until=args.kappa_dense_until,
        )
        summary_path = Path(args.out_dir) / "kappa_summary.json"
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Saved kappa summary to {summary_path}")

        print_kappa_summary(summary, args)

        try:
            plot_kappa(kappa_logger.log, kappa_logger.shape_key_to_name,
                       Path(args.out_dir) / "kappa_trajectory.png")
            print(f"Saved plot to {Path(args.out_dir) / 'kappa_trajectory.png'}")
        except Exception as e:
            print(f"Could not save plot: {e}")

    # ===== Console summary =====
    print("\n========== Final validation losses ==========")
    for opt, h in all_history.items():
        if h["val_loss"]:
            print(f"  {opt:6s}: {h['val_loss'][-1]:.4f}  "
                  f"(wall {h['wall_time'][-1]/3600:.1f}h)")


def compute_kappa_summary(kappa_log_dict, shape_key_to_name, kappa_dense_until=200):
    all_kappa = []
    post_warmup_all = []
    per_param = {}
    for shape_key, traj in kappa_log_dict.items():
        finite = [(e["step"], e["kappa"]) for e in traj
                  if isinstance(e["kappa"], (int, float))
                  and e["kappa"] is not None
                  and not math.isnan(e["kappa"])
                  and not math.isinf(e["kappa"])]
        if not finite:
            continue
        steps_kappas = sorted(finite)
        kappas = [k for _, k in steps_kappas]
        all_kappa.extend(kappas)
        post_warmup_kappas = [k for s, k in steps_kappas if s > kappa_dense_until]
        post_warmup_all.extend(post_warmup_kappas)
        name = shape_key_to_name.get(shape_key, shape_key)
        per_param[name] = {
            "shape_key": shape_key,
            "max": float(max(kappas)),
            "median": float(sorted(kappas)[len(kappas)//2]),
            "p95": float(sorted(kappas)[int(0.95*len(kappas))]) if len(kappas) > 5 else float(max(kappas)),
            "post_warmup_max": float(max(post_warmup_kappas)) if post_warmup_kappas else None,
            "n_observations": len(kappas),
        }
    if not all_kappa:
        return {"overall": {"error": "no finite kappa values"}}

    sorted_k = sorted(all_kappa)
    n = len(sorted_k)
    overall = {
        "n_params": len(per_param),
        "n_total_observations": n,
        "max": float(max(all_kappa)),
        "median": float(sorted_k[n//2]),
        "p95": float(sorted_k[int(0.95*n)]),
        "p99": float(sorted_k[int(0.99*n)]),
    }
    if post_warmup_all:
        sorted_pw = sorted(post_warmup_all)
        npw = len(sorted_pw)
        overall["post_warmup_n"] = npw
        overall["post_warmup_max"] = float(max(post_warmup_all))
        overall["post_warmup_median"] = float(sorted_pw[npw//2])
        overall["post_warmup_p95"] = float(sorted_pw[int(0.95*npw)])
        overall["post_warmup_p99"] = float(sorted_pw[int(0.99*npw)])
    return {"overall": overall, "per_param": per_param}


def print_kappa_summary(summary, args):
    print()
    print("=" * 60)
    print("KAPPA TRACKING SUMMARY")
    print("=" * 60)
    if "error" in summary["overall"]:
        print(f"  Error: {summary['overall']['error']}")
        return
    o = summary["overall"]
    print(f"  Number of Dion-managed matrices: {o['n_params']}")
    print(f"  Total kappa observations:        {o['n_total_observations']}")
    print()
    print(f"  Across all observations:")
    print(f"    max:    {o['max']:.4g}")
    print(f"    median: {o['median']:.4g}")
    print(f"    p95:    {o['p95']:.4g}")
    print(f"    p99:    {o['p99']:.4g}")
    if "post_warmup_max" in o:
        print()
        print(f"  Post-warmup (step > {args.kappa_dense_until}):")
        print(f"    max:    {o['post_warmup_max']:.4g}")
        print(f"    median: {o['post_warmup_median']:.4g}")
        print(f"    p95:    {o['post_warmup_p95']:.4g}")
        print(f"    p99:    {o['post_warmup_p99']:.4g}")
    print()
    print(f"  Reference (Ahn et al. 2025a):")
    print(f"    Reported on 162M GPT: kappa in [200, 1000] post-warmup")
    print(f"    Reported on larger (3B): post-warmup kappa grows with model size (Section A.1)")
    print(f"    Assumed kappa_max in our paper Corollary 1: 10^3")
    if "post_warmup_max" in o:
        verdict = "WITHIN" if o["post_warmup_max"] < 1000 else "EXCEEDS"
        print(f"    Verdict on this 1.3B run: post-warmup max {verdict} kappa_max=10^3")

    sorted_per_param = sorted(
        [(n, p) for n, p in summary["per_param"].items()
         if p.get("post_warmup_max") is not None],
        key=lambda kv: -kv[1]["post_warmup_max"],
    )
    print()
    print("  Top 10 matrices by post-warmup max kappa:")
    print(f"    {'name':40s} {'pw_max':>10s} {'median':>10s} {'p95':>10s}")
    for name, stats in sorted_per_param[:10]:
        short = name if len(name) < 38 else name[:35] + "..."
        print(f"    {short:40s} {stats['post_warmup_max']:>10.3g} "
              f"{stats['median']:>10.3g} {stats['p95']:>10.3g}")


def plot_kappa(kappa_log_dict, shape_key_to_name, save_path, max_lines=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    ax_full, ax_zoom = axes
    cmap = plt.get_cmap("tab20")

    sorted_keys = sorted(kappa_log_dict.keys())
    if len(sorted_keys) > max_lines:
        keyed_by_size = sorted(
            sorted_keys,
            key=lambda k: -math.prod(kappa_log_dict[k][0]["shape"]),
        )
        sorted_keys = keyed_by_size[:max_lines]

    for i, shape_key in enumerate(sorted_keys):
        traj = kappa_log_dict[shape_key]
        steps, kappas = [], []
        for e in traj:
            k = e["kappa"]
            if isinstance(k, (int, float)) and not math.isnan(k) and not math.isinf(k):
                steps.append(e["step"])
                kappas.append(k)
        if not kappas:
            continue
        name = shape_key_to_name.get(shape_key, shape_key)
        label = name if len(name) < 36 else name[:33] + "..."
        ax_full.plot(steps, kappas, label=label, color=cmap(i % 20), lw=0.9, alpha=0.85)
        # zoom: post-warmup
        steps_pw = [s for s in steps if s > 200]
        kappas_pw = [k for s, k in zip(steps, kappas) if s > 200]
        ax_zoom.plot(steps_pw, kappas_pw, color=cmap(i % 20), lw=0.7, alpha=0.6)

    for ax in [ax_full, ax_zoom]:
        ax.axhline(1000, ls="--", color="black", alpha=0.6,
                   label=r"$\kappa=10^3$ (assumed bound)")
        ax.axhline(200, ls=":", color="gray", alpha=0.6,
                   label=r"$\kappa=200$ (Ahn et al. lower)")
        ax.set_xlabel("Step")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")
    ax_full.set_ylabel(r"$\kappa(P_t)$")
    ax_full.set_title(r"$\kappa(P_t)$ along Dion training (1.3B GPT, full trajectory)")
    ax_full.legend(loc="upper right", fontsize=7, ncol=1)
    ax_zoom.set_ylabel(r"$\kappa(P_t)$ (post-warmup)")
    ax_zoom.set_title(r"Post-warmup zoom")
    ax_zoom.set_ylim(10, 2000)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


if __name__ == "__main__":
    main()
