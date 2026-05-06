"""
Experiment: log kappa(B_t) after RCQR pre-whitening during a Dion-RCQR run.
==========================================================================

Validates Lemma 4.4 of the paper: with the random Gaussian sketch and
oversampling c_sk=1.25, the conditioned matrix B = P R1^{-1} that is
fed into the Cholesky-QR step has kappa(B) <= kappa_0 <= 10 with high
probability, IRRESPECTIVE of kappa(P).

This is the empirical content that justifies the claim that RCQR's
orthonormalization error is independent of kappa(P_t).

Strategy
--------
We monkey-patch `optimizers._orthonormalize_rcqr` to record kappa(B)
(after pre-whitening, before Cholesky-QR) and, for context, also kappa(P)
(the un-whitened input). This lets us directly compare:

  - kappa(P_t): the input condition number, Section 5.4 measurement
  - kappa(B_t): the post-whitening condition number, Lemma 4.4 prediction

Output:
  - kappa_B_log.json         per-step kappa(P) and kappa(B) for every matrix
  - kappa_B_summary.json     aggregate stats; key field 'B_max', 'B_p99'
  - kappa_B_trajectory.png   2-panel plot, top: kappa(P), bottom: kappa(B)

The training itself is identical to the kappa_tracking_162m.py setup
but using qr_method=rcqr.
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
# Logger for kappa(P) and kappa(B)
# =============================================================================

class KappaPBLogger:
    """Records both kappa(P) (input) and kappa(B) (after pre-whitening).

    P is the Dion power-iteration input  M @ V       of shape (m, r).
    B = P R1^{-1} is the pre-whitened matrix of shape (m, r) that
    Cholesky-QR is applied to. Lemma 4.4 says kappa(B) <= kappa_0 <= 10
    with high probability over the sketch S.
    """

    def __init__(self):
        # log[shape_key] = list of {step, kappa_P, kappa_B, shape}
        self.log = {}
        self.current_step = 0
        self.recording = True
        self._call_order_in_step = 0
        self.shape_key_to_name = {}

    def reset_step(self, step):
        self.current_step = step
        self._call_order_in_step = 0

    @torch.no_grad()
    def _kappa(self, M):
        """Condition number via SVD."""
        try:
            s = torch.linalg.svdvals(M.float())
            s_max = float(s[0])
            s_min = float(s[-1])
            return s_max / s_min if s_min > 1e-30 else float("inf")
        except Exception:
            return float("nan")

    def record(self, P, B):
        """Called from the patched RCQR after pre-whitening, before Cholesky."""
        if not self.recording:
            self._call_order_in_step += 1
            return
        shape_key = f"shape{tuple(P.shape)}_pos{self._call_order_in_step}"
        self.log.setdefault(shape_key, []).append({
            "step": self.current_step,
            "kappa_P": self._kappa(P),
            "kappa_B": self._kappa(B),
            "shape": list(P.shape),
        })
        self._call_order_in_step += 1

    def to_json_safe(self):
        out = {}
        for k, traj in self.log.items():
            cleaned = []
            for e in traj:
                row = {"step": e["step"], "shape": e["shape"]}
                for fld in ("kappa_P", "kappa_B"):
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
# Monkey-patch RCQR to log kappa(P) and kappa(B)
# =============================================================================

_kappa_pb_logger = None  # set in main()


def _patched_rcqr(P, oversample=1.25, eps=0.0):
    """Identical to _orthonormalize_rcqr but records (kappa(P), kappa(B))."""
    m, r = P.shape
    rt = max(r + 1, int(math.ceil(oversample * r)))
    S = torch.randn(rt, m, device=P.device, dtype=P.dtype) / math.sqrt(rt)
    P_tilde = S @ P
    _, R1 = torch.linalg.qr(P_tilde, mode="reduced")
    B_T = torch.linalg.solve_triangular(R1.T, P.T, upper=False)
    B = B_T.T

    # === LOGGING POINT: P is the input, B is the pre-whitened matrix ===
    if _kappa_pb_logger is not None:
        _kappa_pb_logger.record(P, B)

    G = B.T @ B
    if eps > 0.0:
        G = G + eps * torch.eye(G.size(0), device=G.device, dtype=G.dtype)
    L = torch.linalg.cholesky(G)
    Q_T = torch.linalg.solve_triangular(L, B.T, upper=False)
    return Q_T.T


# To make Dion's _safe_orthonormalize use our patched RCQR, replace
# the module-level reference. Since _safe_orthonormalize calls
# _orthonormalize_rcqr by name, patching that name suffices.

def install_patch():
    optimizers._orthonormalize_rcqr = _patched_rcqr


# =============================================================================
# Pairing shape_key -> param name
# =============================================================================

def build_shape_key_to_name_map(dion_param_names, rank_fraction):
    mapping = {}
    for pos, (name, p) in enumerate(dion_param_names):
        m, n = p.shape
        r = max(1, int(round(rank_fraction * min(m, n))))
        shape_key = f"shape{(m, r)}_pos{pos}"
        mapping[shape_key] = name
    return mapping


# =============================================================================
# Train one Dion-RCQR with kappa(B) tracking
# =============================================================================

def train_with_kappa_B(args, device, logger):
    print("\n========== Training Dion-RCQR with kappa(P), kappa(B) tracking ==========")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    cfg = make_config(args)
    model = GPT(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_param/1e6:.1f}M parameters "
          f"(d_model={cfg.d_model}, layers={cfg.n_layers})")

    matrix_params, _ = split_params(model)
    matrix_param_set = set(id(p) for p in matrix_params)
    dion_param_names = [(name, p) for name, p in model.named_parameters()
                        if id(p) in matrix_param_set]
    print(f"Dion-managed parameters: {len(dion_param_names)}")

    logger.shape_key_to_name = build_shape_key_to_name_map(
        dion_param_names, args.rank_fraction)

    # Force qr_method=rcqr regardless of args
    args_copy = argparse.Namespace(**vars(args))
    args_copy.qr_method = "rcqr"
    args_copy.qr_warmup_steps = 0  # use RCQR from step 1

    m_opt, s_opt = build_optimizers(model, "dion", args_copy)

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

        # Tell logger whether to record this step
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
            n_obs = sum(len(t) for t in logger.log.values())
            print(f"  step {step+1:5d}/{args.steps} | "
                  f"train {loss.item():.4f} | val {val:.4f} | "
                  f"{wall:.1f}s | obs={n_obs}")

    return history


# =============================================================================
# Summary + plot
# =============================================================================

def compute_summary(log, shape_key_to_name, dense_until=0):
    """Aggregate kappa_P and kappa_B across all params/steps."""
    all_P, all_B = [], []
    pw_P, pw_B = [], []
    per_param = {}

    def is_finite(v):
        return (isinstance(v, (int, float)) and v is not None
                and not math.isnan(v) and not math.isinf(v))

    for shape_key, traj in log.items():
        P_vals = [(e["step"], e["kappa_P"]) for e in traj
                  if is_finite(e["kappa_P"])]
        B_vals = [(e["step"], e["kappa_B"]) for e in traj
                  if is_finite(e["kappa_B"])]
        if not P_vals or not B_vals:
            continue
        all_P.extend(v for _, v in P_vals)
        all_B.extend(v for _, v in B_vals)
        pw_P.extend(v for s, v in P_vals if s > dense_until)
        pw_B.extend(v for s, v in B_vals if s > dense_until)
        name = shape_key_to_name.get(shape_key, shape_key)
        per_param[name] = {
            "shape_key": shape_key,
            "P_max": float(max(v for _, v in P_vals)),
            "B_max": float(max(v for _, v in B_vals)),
            "B_median": float(sorted(v for _, v in B_vals)[len(B_vals)//2]),
            "n": len(B_vals),
        }

    if not all_B:
        return {"overall": {"error": "no finite kappa values"}}

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
    overall.update(stats(all_P, "P"))
    overall.update(stats(all_B, "B"))
    if pw_P:
        overall.update(stats(pw_P, "P_postwarmup"))
    if pw_B:
        overall.update(stats(pw_B, "B_postwarmup"))

    return {"overall": overall, "per_param": per_param}


def plot_kappa_PB(log, shape_key_to_name, save_path, max_lines=8):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=(11, 8.5), sharex=True)
    cmap = plt.get_cmap("tab20")
    keys = sorted(log.keys())

    # If too many, prefer largest matrices
    if len(keys) > max_lines:
        keys = sorted(
            keys,
            key=lambda k: -math.prod(log[k][0]["shape"]),
        )[:max_lines]

    for i, key in enumerate(keys):
        traj = log[key]
        steps = [e["step"] for e in traj]
        Pv = [e["kappa_P"] if isinstance(e["kappa_P"], (int, float))
              and not math.isnan(e["kappa_P"]) and not math.isinf(e["kappa_P"])
              else None for e in traj]
        Bv = [e["kappa_B"] if isinstance(e["kappa_B"], (int, float))
              and not math.isnan(e["kappa_B"]) and not math.isinf(e["kappa_B"])
              else None for e in traj]
        sP = [(s, v) for s, v in zip(steps, Pv) if v is not None]
        sB = [(s, v) for s, v in zip(steps, Bv) if v is not None]
        if not sB:
            continue
        name = shape_key_to_name.get(key, key)
        label = name if len(name) < 32 else name[:29] + "..."
        if sP:
            axes[0].plot([s for s, _ in sP], [v for _, v in sP],
                         label=label, color=cmap(i % 20), lw=0.9, alpha=0.85)
        axes[1].plot([s for s, _ in sB], [v for _, v in sB],
                     label=label, color=cmap(i % 20), lw=0.9, alpha=0.85)

    axes[0].axhline(1000, ls="--", color="black", alpha=0.6,
                    label=r"Section 5.4 reported max ($\kappa = 10^3$)")
    axes[0].set_yscale("log")
    axes[0].set_ylabel(r"$\kappa(P_t)$  (input)")
    axes[0].set_title(r"Input $\kappa(P_t)$ — large, strongly varying")
    axes[0].grid(True, alpha=0.3)
    axes[0].legend(fontsize=7, ncol=2, loc="upper right")

    axes[1].axhline(10, ls="--", color="red", alpha=0.7,
                    label=r"Lemma 4.4 prediction ($\kappa_0 = 10$)")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Step")
    axes[1].set_ylabel(r"$\kappa(B_t)$  (post-whitening)")
    axes[1].set_title(r"Post-whitening $\kappa(B_t)$ — bounded by $\kappa_0$ (Lemma 4.4)")
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
    parser.add_argument("--qr_method", default="rcqr")  # forced
    parser.add_argument("--qr_warmup_steps", type=int, default=0)
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--selection", default="l1")
    parser.add_argument("--kappa_stride", type=int, default=50)
    parser.add_argument("--kappa_dense_until", type=int, default=100)
    parser.add_argument("--out_dir", default="results/kappa_B_162m")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    global _kappa_pb_logger
    _kappa_pb_logger = KappaPBLogger()
    install_patch()
    print("Installed kappa(B) tracking patch on _orthonormalize_rcqr")

    history = train_with_kappa_B(args, device, _kappa_pb_logger)

    log_path = Path(args.out_dir) / "kappa_B_log.json"
    with open(log_path, "w") as f:
        json.dump({
            "shape_key_to_name": _kappa_pb_logger.shape_key_to_name,
            "log": _kappa_pb_logger.to_json_safe(),
            "args": vars(args),
        }, f, indent=2)
    print(f"\nSaved kappa(B) log to {log_path}")

    history_path = Path(args.out_dir) / "history.json"
    with open(history_path, "w") as f:
        json.dump({"args": vars(args), "history": {"dion-rcqr": history}},
                  f, indent=2)
    print(f"Saved training history to {history_path}")

    summary = compute_summary(
        _kappa_pb_logger.log,
        _kappa_pb_logger.shape_key_to_name,
        dense_until=args.kappa_dense_until,
    )
    summary_path = Path(args.out_dir) / "kappa_B_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"Saved summary to {summary_path}")

    print()
    print("=" * 70)
    print("KAPPA(B) TRACKING SUMMARY (Dion-RCQR, 162M GPT)")
    print("=" * 70)
    if "error" in summary["overall"]:
        print(f"  Error: {summary['overall']['error']}")
        return
    o = summary["overall"]
    print(f"  Number of Dion-managed matrices: {o['n_params']}")
    print()
    print(f"  kappa(P_t)  (input to RCQR):")
    print(f"    max:    {o.get('P_max', 'n/a'):.4g}")
    print(f"    median: {o.get('P_median', 'n/a'):.4g}")
    print(f"    p99:    {o.get('P_p99', 'n/a'):.4g}")
    print()
    print(f"  kappa(B_t)  (post-whitening, fed to Cholesky-QR):")
    print(f"    max:    {o.get('B_max', 'n/a'):.4g}")
    print(f"    median: {o.get('B_median', 'n/a'):.4g}")
    print(f"    p99:    {o.get('B_p99', 'n/a'):.4g}")
    if "B_postwarmup_max" in o:
        print()
        print(f"  Post-warmup (step > {args.kappa_dense_until}):")
        print(f"    kappa(P) max: {o['P_postwarmup_max']:.4g}")
        print(f"    kappa(B) max: {o['B_postwarmup_max']:.4g}")
    print()
    print(f"  Lemma 4.4 prediction:  kappa(B) <= kappa_0 <= 10")
    if "B_max" in o:
        verdict = "WITHIN" if o["B_max"] <= 10 else "EXCEEDS"
        print(f"  Verdict: {verdict} the kappa_0 = 10 prediction")

    plot_path = Path(args.out_dir) / "kappa_B_trajectory.png"
    try:
        plot_kappa_PB(_kappa_pb_logger.log,
                      _kappa_pb_logger.shape_key_to_name, plot_path)
        print(f"\nSaved plot to {plot_path}")
    except Exception as e:
        print(f"\nCould not save plot: {e}")


if __name__ == "__main__":
    main()
