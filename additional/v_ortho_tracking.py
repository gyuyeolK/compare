"""
Experiment: log ||V_t^T V_t - I||_op during a Dion run, for the variant
that uses ColNorm (the actual production implementation).
========================================================================

Validates the assumption in Section 2.1 that V_t has orthonormal columns
(i.e., ||V_t||_op = 1).

The implementation in optimizers.py uses:
    V_new = W / ||W||_col

This produces V with unit-norm columns, but NOT necessarily orthonormal
(columns can be correlated). For the analysis to apply with constants
matching the orthonormal-V assumption, the columns of V must be close to
orthogonal.

The empirical test: log

    delta_t := ||V_t^T V_t - I||_op

at every Dion step for every Dion-managed matrix. If delta_t stays small
(say <= 0.1) along training, the orthonormal-V assumption is justified
in practice. If delta_t is large (e.g. > 0.5), the analysis assumes a
different algorithm than the implementation runs.

Strategy
--------
We monkey-patch the Dion.step() method to log ||V^T V - I||_op for each
parameter, after V has been updated. We also optionally log
||V||_op (operator norm of V), which is bounded by 1 + delta if V has
unit-norm columns and small column correlation.

Output:
    delta_log.json         per-step delta_t for every matrix
    delta_summary.json     aggregate stats
    delta_trajectory.png   plot, log y-axis
"""

import argparse
import json
import math
import time
from pathlib import Path

import torch

import optimizers
from optimizers import Muon, Dion, Dion2, split_params
from model import GPT, GPTConfig
from data import get_loader
from train_compare import (
    make_config, build_optimizers, lr_schedule, evaluate
)


# =============================================================================
# Logger
# =============================================================================

class VOrthoLogger:
    """Records ||V^T V - I||_op and ||V||_op per Dion-managed param per step."""

    def __init__(self):
        # log[shape_key] = list of {step, delta, V_op, shape}
        self.log = {}
        self.current_step = 0
        self.recording = True
        self._call_order_in_step = 0
        self.shape_key_to_name = {}

    def reset_step(self, step):
        self.current_step = step
        self._call_order_in_step = 0

    @torch.no_grad()
    def measure(self, V):
        """Compute (||V^T V - I||_op, ||V||_op) for V of shape (n, r)."""
        try:
            V32 = V.float()
            r = V32.shape[1]
            G = V32.T @ V32
            I = torch.eye(r, device=V.device, dtype=torch.float32)
            # operator norm of (G - I) = max |eigenvalue|
            eigs = torch.linalg.eigvalsh(G - I)
            delta = float(eigs.abs().max())
            # ||V||_op = sqrt(largest eigenvalue of V^T V)
            eg = torch.linalg.eigvalsh(G)
            V_op = float(eg.max().clamp_min(0).sqrt())
            return delta, V_op
        except Exception:
            return float("nan"), float("nan")

    def record(self, V):
        if not self.recording:
            self._call_order_in_step += 1
            return
        shape_key = f"shape{tuple(V.shape)}_pos{self._call_order_in_step}"
        delta, V_op = self.measure(V)
        self.log.setdefault(shape_key, []).append({
            "step": self.current_step,
            "delta": delta,
            "V_op": V_op,
            "shape": list(V.shape),
        })
        self._call_order_in_step += 1

    def to_json_safe(self):
        out = {}
        for k, traj in self.log.items():
            cleaned = []
            for e in traj:
                row = {"step": e["step"], "shape": e["shape"]}
                for fld in ("delta", "V_op"):
                    v = e[fld]
                    if isinstance(v, float) and math.isnan(v):
                        v = None
                    elif isinstance(v, float) and math.isinf(v):
                        v = "inf"
                    row[fld] = v
                cleaned.append(row)
            out[k] = cleaned
        return out


# =============================================================================
# Patched Dion.step
# =============================================================================

_v_logger = None


def make_patched_dion_step():
    """Build a patched version of Dion.step that logs V at the end of each
    parameter update. Uses the original implementation verbatim, only
    inserts a logging hook after V_new is assigned to state."""
    original_step = Dion.step

    @torch.no_grad()
    def patched_step(self, closure=None):
        # We can't easily insert hooks into the original step; instead,
        # call the original step then read the updated V from state.
        loss = original_step(self, closure)

        if _v_logger is None:
            return loss

        # After the step, state[p]["V"] holds V_new for each param.
        # We iterate over groups in the same order as Dion.step does, so the
        # _call_order_in_step indexing matches the optimizer's internal order.
        _v_logger._call_order_in_step = 0  # reset position counter
        for group in self.param_groups:
            for p in group["params"]:
                if p.ndim != 2:
                    continue
                state = self.state.get(p, None)
                if state is None or "V" not in state:
                    continue
                V = state["V"]
                _v_logger.record(V)
        return loss

    return patched_step


def install_patch():
    Dion.step = make_patched_dion_step()


# =============================================================================
# Pairing
# =============================================================================

def build_shape_key_to_name_map(dion_param_names, rank_fraction):
    mapping = {}
    for pos, (name, p) in enumerate(dion_param_names):
        m, n = p.shape
        r = max(1, int(round(rank_fraction * min(m, n))))
        # V has shape (n, r) -- different from P (which has shape (m, r))
        shape_key = f"shape{(n, r)}_pos{pos}"
        mapping[shape_key] = name
    return mapping


# =============================================================================
# Train one Dion with V-orthonormality tracking
# =============================================================================

def train_with_v_tracking(args, device, logger):
    print("\n========== Training Dion with ||V^T V - I||_op tracking ==========")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    cfg = make_config(args)
    model = GPT(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_param/1e6:.1f}M parameters")

    matrix_params, _ = split_params(model)
    matrix_param_set = set(id(p) for p in matrix_params)
    dion_param_names = [(name, p) for name, p in model.named_parameters()
                        if id(p) in matrix_param_set]
    print(f"Dion-managed parameters: {len(dion_param_names)}")
    for name, p in dion_param_names[:3]:
        n = p.shape[1]
        r = max(1, int(round(args.rank_fraction * min(p.shape))))
        print(f"  {name}: V shape ({n}, {r})")

    logger.shape_key_to_name = build_shape_key_to_name_map(
        dion_param_names, args.rank_fraction)

    m_opt, s_opt = build_optimizers(model, "dion", args)

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

        logger.reset_step(step + 1)
        logger.recording = (
            (step + 1) % args.v_stride == 0
            or step + 1 <= args.v_dense_until
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
            n_obs = sum(len(t) for t in logger.log.values())
            print(f"  step {step+1:5d}/{args.steps} | "
                  f"train {loss.item():.4f} | val {val:.4f} | "
                  f"{wall:.1f}s | obs={n_obs}")

    return history


# =============================================================================
# Summary + plot
# =============================================================================

def compute_summary(log, shape_key_to_name, dense_until=0):
    all_d, all_v = [], []
    pw_d, pw_v = [], []
    per_param = {}

    def is_finite(v):
        return (isinstance(v, (int, float)) and v is not None
                and not math.isnan(v) and not math.isinf(v))

    for shape_key, traj in log.items():
        d_pairs = [(e["step"], e["delta"]) for e in traj
                   if is_finite(e["delta"])]
        v_pairs = [(e["step"], e["V_op"]) for e in traj
                   if is_finite(e["V_op"])]
        if not d_pairs:
            continue
        all_d.extend(v for _, v in d_pairs)
        all_v.extend(v for _, v in v_pairs)
        pw_d.extend(v for s, v in d_pairs if s > dense_until)
        pw_v.extend(v for s, v in v_pairs if s > dense_until)
        name = shape_key_to_name.get(shape_key, shape_key)
        d_vals = [v for _, v in d_pairs]
        per_param[name] = {
            "shape_key": shape_key,
            "delta_max": float(max(d_vals)),
            "delta_median": float(sorted(d_vals)[len(d_vals)//2]),
            "delta_first": d_vals[0],
            "delta_last": d_vals[-1],
            "n": len(d_vals),
        }

    if not all_d:
        return {"overall": {"error": "no finite delta values"}}

    def stats(arr, name):
        if not arr:
            return {}
        s = sorted(arr)
        n = len(s)
        return {
            f"{name}_max": float(max(arr)),
            f"{name}_median": float(s[n//2]),
            f"{name}_p95": float(s[int(0.95 * n)]),
            f"{name}_p99": float(s[int(0.99 * n)]),
            f"{name}_n": n,
        }

    overall = {"n_params": len(per_param)}
    overall.update(stats(all_d, "delta"))
    overall.update(stats(all_v, "V_op"))
    if pw_d:
        overall.update(stats(pw_d, "delta_postwarmup"))
    if pw_v:
        overall.update(stats(pw_v, "V_op_postwarmup"))
    return {"overall": overall, "per_param": per_param}


def plot_v_ortho(log, shape_key_to_name, save_path, max_lines=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    cmap = plt.get_cmap("tab20")
    keys = sorted(log.keys())
    if len(keys) > max_lines:
        keys = sorted(keys,
                      key=lambda k: -math.prod(log[k][0]["shape"]))[:max_lines]

    for i, key in enumerate(keys):
        traj = log[key]
        steps = [e["step"] for e in traj]
        d_pairs = [(s, e["delta"]) for s, e in zip(steps, traj)
                   if isinstance(e["delta"], (int, float))
                   and not math.isnan(e["delta"]) and not math.isinf(e["delta"])]
        v_pairs = [(s, e["V_op"]) for s, e in zip(steps, traj)
                   if isinstance(e["V_op"], (int, float))
                   and not math.isnan(e["V_op"]) and not math.isinf(e["V_op"])]
        if not d_pairs:
            continue
        name = shape_key_to_name.get(key, key)
        label = name if len(name) < 32 else name[:29] + "..."
        axes[0].plot([s for s, _ in d_pairs], [v for _, v in d_pairs],
                     label=label, color=cmap(i % 20), lw=0.9, alpha=0.85)
        if v_pairs:
            axes[1].plot([s for s, _ in v_pairs], [v for _, v in v_pairs],
                         label=label, color=cmap(i % 20), lw=0.9, alpha=0.85)

    axes[0].axhline(0.1, ls="--", color="green", alpha=0.7,
                    label=r"loose threshold ($\delta = 0.1$)")
    axes[0].axhline(0.5, ls="--", color="red", alpha=0.6,
                    label=r"strict threshold ($\delta = 0.5$)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\|V_t^\top V_t - I\|_{\mathrm{op}}$")
    axes[0].set_title(r"Section 2.1 assumption: orthonormality of $V_t$")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")

    axes[1].axhline(1.0, ls="--", color="black", alpha=0.6,
                    label=r"$\|V\|_{\mathrm{op}} = 1$ (orthonormal)")
    axes[1].axhline(math.sqrt(192), ls=":", color="gray", alpha=0.5,
                    label=r"$\sqrt{r}$ (worst case for $\|V\|_{\mathrm{col}}=1$)")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(r"$\|V_t\|_{\mathrm{op}}$")
    axes[1].set_yscale("log")
    axes[1].grid(True, alpha=0.3)
    axes[1].legend(fontsize=7, loc="upper right")

    fig.tight_layout()
    fig.savefig(save_path, dpi=140)
    plt.close(fig)


# =============================================================================
# Main
# =============================================================================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", choices=["synthetic", "fineweb"],
                        default="synthetic")
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--seq_len", type=int, default=1024)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--mlp_mult", type=int, default=4)
    parser.add_argument("--steps", type=int, default=3000)
    parser.add_argument("--warmup_frac", type=float, default=0.0)
    parser.add_argument("--cooldown_frac", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=100)
    parser.add_argument("--eval_batches", type=int, default=10)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--adam_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--rank_fraction", type=float, default=0.25)
    parser.add_argument("--qr_method", choices=["qr", "cholesky", "rcqr"],
                        default="qr")
    parser.add_argument("--qr_warmup_steps", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--selection", default="l1")
    parser.add_argument("--v_stride", type=int, default=50,
                        help="record V every this many steps (post-warmup)")
    parser.add_argument("--v_dense_until", type=int, default=100)
    parser.add_argument("--out_dir", default="results/v_ortho_162m")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    global _v_logger
    _v_logger = VOrthoLogger()
    install_patch()
    print("Installed V-orthonormality tracking patch on Dion.step")

    history = train_with_v_tracking(args, device, _v_logger)

    log_path = Path(args.out_dir) / "delta_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "shape_key_to_name": _v_logger.shape_key_to_name,
            "log": _v_logger.to_json_safe(),
            "args": vars(args),
        }, f, indent=2)
    print(f"\nSaved delta log to {log_path}")

    history_path = Path(args.out_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump({"args": vars(args), "history": {"dion": history}},
                  f, indent=2)
    print(f"Saved training history to {history_path}")

    summary = compute_summary(
        _v_logger.log, _v_logger.shape_key_to_name,
        dense_until=args.v_dense_until,
    )
    summary_path = Path(args.out_dir) / "delta_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    print()
    print("=" * 70)
    print("V-ORTHONORMALITY TRACKING SUMMARY (Dion, 162M GPT)")
    print("=" * 70)
    if "error" in summary["overall"]:
        print(f"  Error: {summary['overall']['error']}")
        return
    o = summary["overall"]
    print(f"  Number of Dion-managed matrices: {o['n_params']}")
    print()
    print(f"  delta_t = ||V_t^T V_t - I||_op:")
    print(f"    max:    {o.get('delta_max', 'n/a'):.4g}")
    print(f"    median: {o.get('delta_median', 'n/a'):.4g}")
    print(f"    p95:    {o.get('delta_p95', 'n/a'):.4g}")
    print(f"    p99:    {o.get('delta_p99', 'n/a'):.4g}")
    print()
    print(f"  ||V_t||_op:")
    print(f"    max:    {o.get('V_op_max', 'n/a'):.4g}")
    print(f"    median: {o.get('V_op_median', 'n/a'):.4g}")
    print(f"    p99:    {o.get('V_op_p99', 'n/a'):.4g}")
    if "delta_postwarmup_max" in o:
        print()
        print(f"  Post-warmup (step > {args.v_dense_until}):")
        print(f"    delta max:   {o['delta_postwarmup_max']:.4g}")
        print(f"    ||V||_op max: {o['V_op_postwarmup_max']:.4g}")
    print()
    print(f"  Section 2.1 assumption: ||V_t||_op = 1 (orthonormal columns)")
    if "delta_max" in o:
        if o["delta_max"] <= 0.1:
            verdict = "STRONGLY VALIDATED (delta <= 0.1)"
        elif o["delta_max"] <= 0.5:
            verdict = "WEAKLY VALIDATED (delta in (0.1, 0.5])"
        else:
            verdict = "VIOLATED (delta > 0.5)"
        print(f"  Verdict: {verdict}")

    plot_path = Path(args.out_dir) / "delta_trajectory.png"
    try:
        plot_v_ortho(_v_logger.log, _v_logger.shape_key_to_name, plot_path)
        print(f"\nSaved plot to {plot_path}")
    except Exception as e:
        print(f"\nCould not save plot: {e}")


if __name__ == "__main__":
    main()
