"""
Kappa(P_t) tracking experiment for Dion on the 162M GPT model.
==============================================================

This script directly tests the assumption sup_t kappa(P_t) <= kappa_max
that underlies Corollary 1 (Dion-CQR concrete bound) of our paper.

Designed to integrate with the existing 162M experiment code base
without modifying any of the original files (optimizers.py, model.py,
data.py, train_compare.py).

Strategy
--------
We monkey-patch `optimizers._safe_orthonormalize` to also record
kappa(P_t) = sigma_max(P_t) / sigma_min(P_t) for each call. The patched
function records into a global log keyed by (param_id, step).

To keep the SVD overhead manageable on a 162M model with ~50 Dion-managed
matrices and 3000 steps, by default we measure only every `--kappa_stride`
steps (default 50). This gives ~60 timestamps per matrix, plenty to
characterise the trajectory.

Usage
-----
    # GPU run with default settings (162M, 3000 steps, stride=50)
    python kappa_tracking_162m.py \\
        --data fineweb \\
        --d_model 768 --n_layers 12 --n_heads 12 \\
        --batch_size 8 --seq_len 1024 \\
        --steps 3000 --log_every 100 \\
        --lr 0.01 --adam_lr 3e-4 \\
        --rank_fraction 0.25 \\
        --qr_method qr --qr_warmup_steps 0 \\
        --kappa_stride 50 \\
        --out_dir results/kappa_162m

    # Cheaper variant (stride 100, no full SVD - power iteration estimate)
    python kappa_tracking_162m.py [...same...] --kappa_stride 100 --kappa_method power_iter

Outputs
-------
    {out_dir}/kappa_log.json        per-step kappa for every matrix
    {out_dir}/kappa_summary.json    aggregate statistics
    {out_dir}/kappa_trajectory.png  trajectory plot, log y-axis
    {out_dir}/history.json          standard training history (loss curves)

Implementation note
-------------------
The kappa values do NOT enter the Dion update -- they are only logged.
Training behaviour is identical to running train_compare.py with the same
arguments, modulo a small wall-clock overhead for the SVDs.
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

# Import the existing 162M code base unchanged.
import optimizers   # the ORIGINAL optimizers.py
from optimizers import Muon, Dion, Dion2, split_params
from model import GPT, GPTConfig
from data import get_loader
from train_compare import (
    make_config, build_optimizers, lr_schedule, evaluate
)


# =============================================================================
# Kappa logger (global, populated via the monkey-patch below)
# =============================================================================

class KappaLogger:
    """Records kappa(P_t) for every Dion `_safe_orthonormalize` call.

    The Dion optimizer calls `_safe_orthonormalize(P, method)` once per
    matrix per step. We patch this function to also record kappa(P)
    into our log.

    Key for each entry: (id(P)-derived identifier we resolve later).
    Since identifying P by Python id() is unreliable across calls (P is
    a fresh tensor each step), we instead use shape + a per-matrix
    counter. The pairing back to parameter names is done after the run.
    """

    def __init__(self, kappa_method="svd", record_first_k=None):
        # log[shape_key][step] = kappa
        self.log = {}
        # global step counter (updated externally)
        self.current_step = 0
        # whether to record on this step (set externally based on stride)
        self.recording = True
        # how to compute kappa
        self.kappa_method = kappa_method
        # for shape collisions (multiple matrices with same shape), distinguish
        # by call-order within a step
        self._call_order_in_step = 0
        # max number of distinct (shape, position) keys to record (None = all)
        self.record_first_k = record_first_k
        # Pairing: shape_key -> param_name (filled once we know names)
        self.shape_key_to_name = {}

    def reset_step(self, step):
        self.current_step = step
        self._call_order_in_step = 0

    @torch.no_grad()
    def measure_kappa(self, P):
        """Compute condition number of P. Returns float (or None on failure)."""
        try:
            if self.kappa_method == "svd":
                # full SVD on (m, r). Cost ~ O(m r^2). For r=192, m=3072,
                # ~1e8 FLOPS, sub-millisecond on H100.
                svals = torch.linalg.svdvals(P.float())
                s_max = float(svals[0])
                s_min = float(svals[-1])
            elif self.kappa_method == "power_iter":
                # cheap estimate: top-2 singular values via 5-step power iter on
                # P^T P. Underestimates kappa (true sigma_min may be smaller),
                # but stable upper-end estimate is what we care about for
                # validating kappa_max < 10^3.
                G = P.T @ P  # (r, r)
                e = torch.linalg.eigvalsh(G.float())  # ascending
                # e are eigenvalues of P^T P = singular values squared
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
        """Called for each P that goes into _safe_orthonormalize."""
        if not self.recording:
            self._call_order_in_step += 1
            return
        # shape key + call order distinguishes same-shape matrices
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
                cleaned.append({
                    "step": entry["step"],
                    "kappa": v,
                    "shape": entry["shape"],
                })
            out[k] = cleaned
        return out


# =============================================================================
# Monkey-patch: wrap _safe_orthonormalize with kappa logging
# =============================================================================

_kappa_logger_global = None  # set by main() before training


_original_safe_orthonormalize = optimizers._safe_orthonormalize


def _patched_safe_orthonormalize(P, method):
    """Monkey-patched version that logs kappa(P) before orthonormalising."""
    if _kappa_logger_global is not None:
        _kappa_logger_global.record(P)
    return _original_safe_orthonormalize(P, method)


# =============================================================================
# Pairing kappa-log shape-keys to parameter names
# =============================================================================

def build_shape_key_to_name_map(model, dion_param_names):
    """Map from shape_key (used during the run) to human-readable param name.

    Within a Dion step the optimizer iterates over its param_groups in a
    deterministic order. _safe_orthonormalize is called once per parameter
    in that order. We rebuild the same iteration order here and pair it
    with the call-order index used in KappaLogger.

    `dion_param_names` is a list of (name, parameter) in the same order
    that the Dion optimizer iterates over them.
    """
    mapping = {}
    for pos, (name, param) in enumerate(dion_param_names):
        m, n = param.shape
        r = max(1, int(round(0.25 * min(m, n))))  # default rank_fraction
        # P = M @ V has shape (m, r)
        shape_key = f"shape{(m, r)}_pos{pos}"
        mapping[shape_key] = name
    return mapping


# =============================================================================
# Train one optimizer with kappa tracking
# =============================================================================

def train_one_with_kappa(opt_name, args, device, logger):
    print(f"\n========== Training with {opt_name.upper()} (with kappa tracking) ==========")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    cfg = make_config(args)
    model = GPT(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_param/1e6:.1f}M parameters "
          f"(d_model={cfg.d_model}, layers={cfg.n_layers})")

    # Identify Dion-managed parameters in the order Dion iterates over them.
    # build_optimizers calls split_params(model), so we replicate that order.
    matrix_params, _ = split_params(model)
    matrix_param_set = set(id(p) for p in matrix_params)
    dion_param_names = [(name, p) for name, p in model.named_parameters()
                        if id(p) in matrix_param_set]
    print(f"Dion-managed parameters: {len(dion_param_names)}")
    for name, p in dion_param_names[:5]:
        print(f"  {name}: shape {tuple(p.shape)}")
    if len(dion_param_names) > 5:
        print(f"  ... and {len(dion_param_names)-5} more")

    # Pre-populate the shape-key -> name map (for output)
    logger.shape_key_to_name = build_shape_key_to_name_map(
        model, dion_param_names)

    m_opt, s_opt = build_optimizers(model, opt_name, args)

    train_iter = get_loader(args.data, args.batch_size, args.seq_len,
                            cfg.vocab_size, device, seed=args.seed)
    val_iter = get_loader(args.data, args.batch_size, args.seq_len,
                          cfg.vocab_size, device, seed=args.seed + 1)

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

        xb, yb = next(train_iter)

        with torch.amp.autocast(
                device_type="cuda" if device.type == "cuda" else "cpu",
                dtype=torch.bfloat16,
                enabled=(device.type == "cuda")):
            _, loss = model(xb, yb)

        m_opt.zero_grad(set_to_none=True)
        s_opt.zero_grad(set_to_none=True)
        loss.backward()

        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in s_opt.param_groups for p in g["params"]],
                args.grad_clip,
            )

        # === KAPPA TRACKING: tell the logger whether to record this step ===
        logger.reset_step(step + 1)
        logger.recording = (
            (step + 1) % args.kappa_stride == 0
            or step + 1 <= args.kappa_dense_until
        )

        m_opt.step()
        s_opt.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            wall = time.time() - t_start
            val = evaluate(model, val_iter, device,
                           num_batches=args.eval_batches)
            history["step"].append(step + 1)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val)
            history["wall_time"].append(wall)
            n_kappa_recorded = sum(len(t) for t in logger.log.values())
            print(f"  step {step+1:5d}/{args.steps} | "
                  f"train {loss.item():.4f} | val {val:.4f} | "
                  f"lr_mult {sched:.3f} | {wall:.1f}s | "
                  f"kappa_obs={n_kappa_recorded}")

    return history


# =============================================================================
# Summary statistics + plot
# =============================================================================

def compute_summary(kappa_log_dict, shape_key_to_name, kappa_dense_until=0):
    """Aggregate kappa statistics across all params and steps.

    kappa_dense_until: steps <= this are considered 'pre-warmup' and
    excluded from post-warmup statistics.
    """
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
            "p95": float(sorted(kappas)[int(0.95*len(kappas))]),
            "first": kappas[0],
            "last": kappas[-1],
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
        "p90": float(sorted_k[int(0.90*n)]),
        "p95": float(sorted_k[int(0.95*n)]),
        "p99": float(sorted_k[int(0.99*n)]),
        "mean": float(sum(all_kappa) / n),
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


def plot_kappa(kappa_log_dict, shape_key_to_name, save_path,
               max_lines=12):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11, 5.5))
    cmap = plt.get_cmap("tab20")
    sorted_keys = sorted(kappa_log_dict.keys())

    # If too many matrices, prefer largest (by shape) to plot
    if len(sorted_keys) > max_lines:
        keyed_by_size = sorted(
            sorted_keys,
            key=lambda k: -math.prod(kappa_log_dict[k][0]["shape"]),
        )
        sorted_keys = keyed_by_size[:max_lines]
        n_skipped = len(kappa_log_dict) - max_lines
        print(f"Plotting {max_lines} largest matrices (skipping {n_skipped} smaller)")

    for i, shape_key in enumerate(sorted_keys):
        traj = kappa_log_dict[shape_key]
        steps = []
        kappas = []
        for e in traj:
            k = e["kappa"]
            if (isinstance(k, (int, float))
                    and not math.isnan(k) and not math.isinf(k)):
                steps.append(e["step"])
                kappas.append(k)
        if not kappas:
            continue
        name = shape_key_to_name.get(shape_key, shape_key)
        # truncate long names
        label = name if len(name) < 36 else name[:33] + "..."
        ax.plot(steps, kappas, label=label, color=cmap(i % 20),
                lw=0.9, alpha=0.85)

    # Reference lines from Section 4.1 of the paper
    ax.axhline(1000, ls="--", color="black", alpha=0.6,
               label=r"Ahn et al. upper estimate ($\kappa = 10^3$)")
    ax.axhline(200, ls=":", color="gray", alpha=0.6,
               label=r"Ahn et al. lower estimate ($\kappa = 200$)")

    ax.set_xlabel("Step")
    ax.set_ylabel(r"$\kappa(P_t)$")
    ax.set_yscale("log")
    ax.set_title(r"$\kappa(P_t)$ along Dion training (162M GPT, FineWeb-Edu)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper right", fontsize=7, ncol=2)
    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


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
    parser.add_argument("--batch_size", type=int, default=8)
    # model (162M defaults: d_model=768, n_layers=12, n_heads=12)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--mlp_mult", type=int, default=4)
    # training
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup_frac", type=float, default=0.0)
    parser.add_argument("--cooldown_frac", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    # optimizer (we always use Dion for the kappa experiment)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--rank_fraction", type=float, default=0.25)
    parser.add_argument("--qr_method", choices=["qr", "cholesky", "rcqr"],
                        default="qr")
    parser.add_argument("--qr_warmup_steps", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.25)  # unused for dion
    parser.add_argument("--selection", choices=["l1", "random"],
                        default="l1")  # unused for dion
    # kappa tracking
    parser.add_argument("--kappa_stride", type=int, default=50,
                        help="record kappa(P_t) every this many steps "
                             "(after the dense-warmup window). Default 50 "
                             "gives ~60 timestamps per matrix at 3000 steps.")
    parser.add_argument("--kappa_dense_until", type=int, default=100,
                        help="record kappa(P_t) at every step until this "
                             "many steps elapsed. Captures the warmup "
                             "transient. Default 100.")
    parser.add_argument("--kappa_method", choices=["svd", "power_iter"],
                        default="svd",
                        help="how to compute kappa(P). 'svd' uses full SVD "
                             "(exact). 'power_iter' uses eigenvalues of "
                             "P^T P (cheaper, slight underestimate of true "
                             "kappa).")
    parser.add_argument("--kappa_record_first_k", type=int, default=None,
                        help="if set, only record kappa for the first k "
                             "shape-distinct matrices. Use to limit memory "
                             "/ overhead.")
    # output
    parser.add_argument("--out_dir", default="results/kappa_162m")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    if device.type == "cuda":
        print(f"GPU: {torch.cuda.get_device_name(0)}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    # === Install the kappa-logging monkey-patch ===
    global _kappa_logger_global
    logger = KappaLogger(
        kappa_method=args.kappa_method,
        record_first_k=args.kappa_record_first_k,
    )
    _kappa_logger_global = logger
    optimizers._safe_orthonormalize = _patched_safe_orthonormalize
    print(f"Installed kappa-logging patch (method={args.kappa_method}, "
          f"stride={args.kappa_stride}, dense_until={args.kappa_dense_until})")

    # === Run training with kappa tracking ===
    history = train_one_with_kappa("dion", args, device, logger)

    # === Save kappa log ===
    log_path = Path(args.out_dir) / "kappa_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "shape_key_to_name": logger.shape_key_to_name,
            "log": logger.to_json_safe(),
            "args": vars(args),
        }, f, indent=2)
    print(f"\nSaved kappa log to {log_path}")

    # === Save training history ===
    history_path = Path(args.out_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump({"args": vars(args), "history": {"dion": history}}, f, indent=2)
    print(f"Saved training history to {history_path}")

    # === Compute summary statistics ===
    summary = compute_summary(
        logger.log, logger.shape_key_to_name,
        kappa_dense_until=args.kappa_dense_until,
    )
    summary_path = Path(args.out_dir) / "kappa_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    # === Print summary to console ===
    print()
    print("=" * 60)
    print("KAPPA TRACKING SUMMARY (162M GPT, Dion)")
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
    print(f"  Reference (Ahn et al. 2025a, Sec. A.1):")
    print(f"    Reported range on 162M GPT: [200, 1000]")
    print(f"    Assumed kappa_max in our paper: 10^3")
    if "post_warmup_max" in o:
        verdict = "WITHIN" if o["post_warmup_max"] < 1000 else "EXCEEDS"
        print(f"    Verdict: post-warmup max {verdict} the assumed kappa_max")

    # === Per-param breakdown (top 10 by max kappa) ===
    sorted_per_param = sorted(
        summary["per_param"].items(),
        key=lambda kv: -kv[1]["max"],
    )
    print()
    print("  Top 10 matrices by max kappa:")
    print(f"    {'name':40s} {'max':>10s} {'median':>10s} {'p95':>10s}")
    for name, stats in sorted_per_param[:10]:
        short = name if len(name) < 38 else name[:35] + "..."
        print(f"    {short:40s} {stats['max']:>10.3g} "
              f"{stats['median']:>10.3g} {stats['p95']:>10.3g}")

    # === Plot trajectory ===
    plot_path = Path(args.out_dir) / "kappa_trajectory.png"
    try:
        plot_kappa(logger.log, logger.shape_key_to_name, plot_path)
        print(f"\nSaved plot to {plot_path}")
    except Exception as e:
        print(f"\nCould not save plot: {e}")


if __name__ == "__main__":
    main()
