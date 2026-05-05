"""
Overlay multiple history.json files on a single plot for direct comparison.

Usage:
    # Single panel, x-axis = steps (default)
    python compare_runs.py --runs A.json:A B.json:B --out compare.png

    # Single panel, x-axis = wall-clock time
    python compare_runs.py --runs A.json:A B.json:B --x time --out compare.png

    # Two side-by-side panels: steps on the left, time on the right
    python compare_runs.py --runs A.json:A B.json:B --side_by_side --out compare.png

Each --runs entry is "PATH:LABEL". The optimizer used inside each json
is the one chosen for that label (the script picks the first opt found).
"""

import argparse
import json
import matplotlib.pyplot as plt


def _plot_one(ax, runs, metric, x_axis):
    """Plot all runs on a single axis."""
    for spec in runs:
        path, label = spec.rsplit(":", 1)
        with open(path) as f:
            data = json.load(f)
        opt_name, h = next(iter(data["history"].items()))
        x = h["wall_time"] if x_axis == "time" else h["step"]
        ax.plot(x, h[metric], label=label, alpha=0.85, marker="o", markersize=3)

    ax.set_xlabel("Wall-clock time (s)" if x_axis == "time" else "Step")
    ax.set_ylabel("Validation loss" if metric == "val_loss" else "Train loss")
    title_x = "wall-clock time" if x_axis == "time" else "step"
    ax.set_title(f"{metric} vs {title_x}")
    ax.legend()
    ax.grid(True, alpha=0.3)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--runs", nargs="+", required=True,
                   help="list of PATH:LABEL pairs")
    p.add_argument("--out", default="results/compare.png")
    p.add_argument("--metric", choices=["val_loss", "train_loss"],
                   default="val_loss")
    p.add_argument("--x", choices=["step", "time"], default="step",
                   help="x-axis when plotting a single panel "
                        "(ignored if --side_by_side)")
    p.add_argument("--side_by_side", action="store_true",
                   help="produce two panels: steps on the left, "
                        "wall-clock time on the right")
    p.add_argument("--ylim", nargs=2, type=float, default=None)
    p.add_argument("--xlim", nargs=2, type=float, default=None)
    args = p.parse_args()

    if args.side_by_side:
        fig, axes = plt.subplots(1, 2, figsize=(13, 4.5), sharey=True)
        _plot_one(axes[0], args.runs, args.metric, "step")
        _plot_one(axes[1], args.runs, args.metric, "time")
        if args.ylim:
            axes[0].set_ylim(*args.ylim)
        # share legend: keep only the right one (or just use both)
        # tighten layout
    else:
        fig, ax = plt.subplots(figsize=(7, 4.5))
        _plot_one(ax, args.runs, args.metric, args.x)
        if args.ylim:
            ax.set_ylim(*args.ylim)
        if args.xlim:
            ax.set_xlim(*args.xlim)

    fig.tight_layout()
    fig.savefig(args.out, dpi=150)
    print(f"Saved {args.out}")


if __name__ == "__main__":
    main()
