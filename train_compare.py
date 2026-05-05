"""
Compare Muon, Dion, and Dion2 on a small GPT.

Usage examples
--------------
# Quick smoke test on synthetic data (no network needed):
python train_compare.py --data synthetic --steps 200 --d_model 256 --n_layers 4

# 160M-style model on FineWeb-Edu (needs internet + datasets/transformers):
python train_compare.py --data fineweb --steps 3000 \
    --d_model 768 --n_layers 12 --n_heads 12 --batch_size 8 --seq_len 1024

# Run only one optimizer (for ablations):
python train_compare.py --opts muon dion2 --steps 500
"""

import argparse
import json
import math
import os
import time
from pathlib import Path

import torch

from model import GPT, GPTConfig
from optimizers import Muon, Dion, Dion2, split_params
from data import get_loader


def make_config(args):
    # auto-adjust n_heads if it doesn't divide d_model
    n_heads = args.n_heads
    if args.d_model % n_heads != 0:
        # pick the largest divisor of d_model that is <= requested n_heads
        for h in range(min(n_heads, args.d_model), 0, -1):
            if args.d_model % h == 0:
                print(f"[note] adjusting n_heads {args.n_heads} -> {h} "
                      f"so it divides d_model={args.d_model}")
                n_heads = h
                break
    return GPTConfig(
        vocab_size=args.vocab_size,
        d_model=args.d_model,
        n_layers=args.n_layers,
        n_heads=n_heads,
        seq_len=args.seq_len,
        mlp_mult=args.mlp_mult,
    )


def build_optimizers(model, opt_name, args):
    """Build a (matrix_optimizer, scalar_optimizer) pair.

    matrix_optimizer:  Muon / Dion / Dion2 -- only operates on 2D weight matrices
                       inside the transformer blocks (attention + MLP).
    scalar_optimizer:  AdamW for embeddings, LM head, biases, norms.
    """
    matrix_params, scalar_params = split_params(model)

    if opt_name == "muon":
        m_opt = Muon(matrix_params, lr=args.lr, momentum=0.95, nesterov=True,
                     ns_steps=5, weight_decay=args.weight_decay)
    elif opt_name == "dion":
        m_opt = Dion(matrix_params, lr=args.lr, rank_fraction=args.rank_fraction,
                     beta=0.05, weight_decay=args.weight_decay,
                     qr_method=args.qr_method,
                     qr_warmup_steps=args.qr_warmup_steps)
    elif opt_name == "dion2":
        m_opt = Dion2(matrix_params, lr=args.lr, alpha=args.alpha,
                      momentum_decay=0.95, selection=args.selection,
                      ns_steps=5, weight_decay=args.weight_decay)
    else:
        raise ValueError(f"unknown optimizer: {opt_name}")

    s_opt = torch.optim.AdamW(
        scalar_params,
        lr=args.adam_lr,
        betas=(0.9, 0.95),
        weight_decay=0.0,
    )
    return m_opt, s_opt


def lr_schedule(step, total_steps, warmup_frac=0.0, cooldown_frac=0.1):
    """Trapezoidal: optional linear warmup, flat, then linear cooldown to 0."""
    warmup = int(warmup_frac * total_steps)
    cooldown_start = int((1 - cooldown_frac) * total_steps)
    if step < warmup:
        return (step + 1) / max(warmup, 1)
    if step >= cooldown_start:
        # linear from 1.0 -> 0.0
        remaining = total_steps - step
        total_cool = total_steps - cooldown_start
        return max(remaining / max(total_cool, 1), 0.0)
    return 1.0


def evaluate(model, loader, device, num_batches=20):
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(num_batches):
            try:
                xb, yb = next(loader)
            except StopIteration:
                break
            with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                     dtype=torch.bfloat16,
                                     enabled=(device.type == "cuda")):
                _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    return sum(losses) / max(len(losses), 1)


def train_one(opt_name, args, device):
    print(f"\n========== Training with {opt_name.upper()} ==========")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(args.seed)

    cfg = make_config(args)
    model = GPT(cfg).to(device)
    n_param = sum(p.numel() for p in model.parameters())
    print(f"Model has {n_param/1e6:.1f}M parameters "
          f"(d_model={cfg.d_model}, layers={cfg.n_layers})")

    m_opt, s_opt = build_optimizers(model, opt_name, args)

    train_iter = get_loader(args.data, args.batch_size, args.seq_len,
                            cfg.vocab_size, device,
                            **({"seed": args.seed} if args.data == "synthetic"
                               else {"seed": args.seed}))
    val_iter = get_loader(args.data, args.batch_size, args.seq_len,
                          cfg.vocab_size, device,
                          **({"seed": args.seed + 1} if args.data == "synthetic"
                             else {"seed": args.seed + 1}))

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

        with torch.amp.autocast(device_type="cuda" if device.type == "cuda" else "cpu",
                                 dtype=torch.bfloat16,
                                 enabled=(device.type == "cuda")):
            _, loss = model(xb, yb)

        m_opt.zero_grad(set_to_none=True)
        s_opt.zero_grad(set_to_none=True)
        loss.backward()

        # optional gradient clipping for the AdamW group
        if args.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                [p for g in s_opt.param_groups for p in g["params"]],
                args.grad_clip,
            )

        m_opt.step()
        s_opt.step()

        if (step + 1) % args.log_every == 0 or step == 0:
            wall = time.time() - t_start
            val = evaluate(model, val_iter, device, num_batches=args.eval_batches)
            history["step"].append(step + 1)
            history["train_loss"].append(loss.item())
            history["val_loss"].append(val)
            history["wall_time"].append(wall)
            print(f"  step {step+1:5d}/{args.steps} | "
                  f"train {loss.item():.4f} | val {val:.4f} | "
                  f"lr_mult {sched:.3f} | {wall:.1f}s")

    return history


def main():
    parser = argparse.ArgumentParser()
    # data
    parser.add_argument("--data", choices=["synthetic", "fineweb"], default="synthetic")
    parser.add_argument("--vocab_size", type=int, default=50304)
    parser.add_argument("--seq_len", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=8)
    # model
    parser.add_argument("--d_model", type=int, default=384)
    parser.add_argument("--n_layers", type=int, default=6)
    parser.add_argument("--n_heads", type=int, default=6)
    parser.add_argument("--mlp_mult", type=int, default=4)
    # training
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--warmup_frac", type=float, default=0.0)
    parser.add_argument("--cooldown_frac", type=float, default=0.1)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log_every", type=int, default=50)
    parser.add_argument("--eval_batches", type=int, default=10)
    # optimizers to compare
    parser.add_argument("--opts", nargs="+",
                        default=["muon", "dion", "dion2"],
                        choices=["muon", "dion", "dion2"])
    parser.add_argument("--lr", type=float, default=0.02,
                        help="learning rate for matrix optimizer")
    parser.add_argument("--adam_lr", type=float, default=3e-4,
                        help="learning rate for AdamW (scalar params)")
    parser.add_argument("--weight_decay", type=float, default=0.01)
    # Dion-specific
    parser.add_argument("--rank_fraction", type=float, default=0.25)
    parser.add_argument("--qr_method", choices=["qr", "cholesky", "rcqr"],
                        default="qr",
                        help="orthonormalization method inside Dion's power "
                             "iteration. cholesky/rcqr are much faster but "
                             "may need a warmup; see Dion paper Appendix A.")
    parser.add_argument("--qr_warmup_steps", type=int, default=200,
                        help="number of plain-QR steps before switching to "
                             "qr_method. Set to 0 to use the fast method "
                             "from step 1.")
    # Dion2-specific
    parser.add_argument("--alpha", type=float, default=0.25)
    parser.add_argument("--selection", choices=["l1", "random"], default="l1")
    # output
    parser.add_argument("--out_dir", default="results")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    Path(args.out_dir).mkdir(parents=True, exist_ok=True)

    all_history = {}
    for opt in args.opts:
        all_history[opt] = train_one(opt, args, device)

    # save
    out_path = Path(args.out_dir) / "history.json"
    with open(out_path, "w") as f:
        json.dump({"args": vars(args), "history": all_history}, f, indent=2)
    print(f"\nSaved history to {out_path}")

    # quick console summary
    print("\n========== Final validation losses ==========")
    for opt, h in all_history.items():
        if h["val_loss"]:
            print(f"  {opt:6s}: {h['val_loss'][-1]:.4f}")


if __name__ == "__main__":
    main()
