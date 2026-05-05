"""
Plot loss curves from `results/history.json`.

Usage:
    python plot.py --history results/history.json --out results/curves.png
"""

import argparse
import json
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--history", default="results/history.json")
    p.add_argument("--out", default="results/curves.png")
    args = p.parse_args()

    with open(args.history) as f:
        data = json.load(f)
    hist = data["history"]

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    for opt, h in hist.items():
        axes[0].plot(h["step"], h["train_loss"], label=opt, alpha=0.8)
        axes[1].plot(h["step"], h["val_loss"], label=opt, alpha=0.8)

    axes[0].set_xlabel("Step")
    axes[0].set_ylabel("Train loss")
    axes[0].set_title("Training loss")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].set_xlabel("Step")
    axes[1].set_ylabel("Validation loss")
    axes[1].set_title("Validation loss")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
