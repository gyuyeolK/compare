"""
Wall-clock-per-step benchmark for the matrix update step of
Muon vs Dion vs Dion2, reproducing Figure 1 of the Dion / Dion2 papers
on a single device.

We construct a square (n x n) parameter and a random gradient, then time
the optimizer step in isolation. Communication is excluded (single device).

Usage:
    python benchmark.py --sizes 2048 4096 8192 16384 --warmup 5 --iters 20
"""

import argparse
import time
import torch

from optimizers import Muon, Dion, Dion2


def benchmark_one(make_opt, n, device, warmup=5, iters=20, dtype=torch.float32):
    p = torch.randn(n, n, device=device, dtype=dtype, requires_grad=False)
    p = torch.nn.Parameter(p)
    opt = make_opt([p])
    # Pre-fill .grad
    for _ in range(warmup):
        p.grad = torch.randn_like(p)
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(iters):
        p.grad = torch.randn_like(p)
        opt.step()
    if device.type == "cuda":
        torch.cuda.synchronize()
    dt = (time.time() - t0) / iters
    return dt


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int,
                        default=[2048, 4096, 8192])
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--iters", type=int, default=10)
    parser.add_argument("--rank_fractions", nargs="+", type=float,
                        default=[0.25, 0.0625])  # 1/4, 1/16
    parser.add_argument("--alphas", nargs="+", type=float,
                        default=[0.5, 0.25])
    parser.add_argument("--include_dion_fast", action="store_true",
                        help="also benchmark Dion with cholesky-QR and RCQR "
                             "(skips warmup so the timing reflects the fast "
                             "path from step 1)")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"{'size':>8} {'optimizer':>30} {'time/step (s)':>15}")
    print("-" * 58)

    for n in args.sizes:
        # Muon
        t = benchmark_one(lambda ps: Muon(ps, lr=0.02), n, device,
                          warmup=args.warmup, iters=args.iters)
        print(f"{n:>8} {'Muon':>30} {t:>15.5f}")
        # Dion at various ranks (default: standard QR)
        for rf in args.rank_fractions:
            t = benchmark_one(
                lambda ps, rf=rf: Dion(ps, lr=0.01, rank_fraction=rf,
                                       qr_method="qr", qr_warmup_steps=0),
                n, device, warmup=args.warmup, iters=args.iters)
            print(f"{n:>8} {'Dion qr (rf=' + str(rf) + ')':>30} {t:>15.5f}")
        # Optionally also bench Dion with the fast QR variants
        if args.include_dion_fast:
            for rf in args.rank_fractions:
                t = benchmark_one(
                    lambda ps, rf=rf: Dion(ps, lr=0.01, rank_fraction=rf,
                                           qr_method="cholesky",
                                           qr_warmup_steps=0),
                    n, device, warmup=args.warmup, iters=args.iters)
                print(f"{n:>8} {'Dion chol (rf=' + str(rf) + ')':>30} {t:>15.5f}")
            for rf in args.rank_fractions:
                t = benchmark_one(
                    lambda ps, rf=rf: Dion(ps, lr=0.01, rank_fraction=rf,
                                           qr_method="rcqr",
                                           qr_warmup_steps=0),
                    n, device, warmup=args.warmup, iters=args.iters)
                print(f"{n:>8} {'Dion rcqr (rf=' + str(rf) + ')':>30} {t:>15.5f}")
        # Dion2 at various alphas
        for a in args.alphas:
            t = benchmark_one(
                lambda ps, a=a: Dion2(ps, lr=0.02, alpha=a, selection="l1"),
                n, device, warmup=args.warmup, iters=args.iters)
            print(f"{n:>8} {'Dion2 (a=' + str(a) + ')':>30} {t:>15.5f}")
        print()


if __name__ == "__main__":
    main()
