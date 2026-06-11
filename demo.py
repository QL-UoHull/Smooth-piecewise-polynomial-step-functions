"""
Demonstration of recursive smooth piecewise polynomial step functions.

Compares the recursive piecewise polynomial construction with classical
polynomial smoothstep functions and optionally saves comparison plots.

Usage
-----
    python demo.py
    python demo.py --no-plot         # print table only, no matplotlib required
    python demo.py --orders 1 2 3 4  # choose smoothness orders
"""

import argparse

import numpy as np

from smooth_step.classical import smooth_step as classical_step
from smooth_step.recursive import smooth_step as recursive_step


# ---------------------------------------------------------------------------
# Comparison table
# ---------------------------------------------------------------------------

def print_comparison_table(orders=(1, 2, 3), n_points=9):
    """Print a comparison table of recursive vs classical values."""
    x = np.linspace(0, 1, n_points)

    for n in orders:
        deg_rec = n + 1
        pieces_rec = n + 1
        deg_cls = 2 * n + 1

        print(f"\nOrder {n}  (C^{n} smooth)")
        print(f"  Recursive : degree-{deg_rec} piecewise polynomials, "
              f"{pieces_rec} pieces")
        print(f"  Classical : single degree-{deg_cls} polynomial")
        print()
        print(f"  {'x':>8}  {'Recursive':>12}  {'Classical':>12}  "
              f"{'Difference':>12}")
        print(f"  {'-' * 8}  {'-' * 12}  {'-' * 12}  {'-' * 12}")

        r = recursive_step(x, n)
        c = classical_step(x, n)
        for xi, ri, ci in zip(x, r, c):
            print(f"  {xi:>8.4f}  {ri:>12.6f}  {ci:>12.6f}  {ci - ri:>+12.6f}")


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison(orders=(1, 2, 3), output_path="smooth_step_comparison.png"):
    """Plot recursive vs classical smooth step functions and save to file."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is not installed; skipping plots.")
        print("Install it with:  pip install matplotlib")
        return

    x = np.linspace(0, 1, 500)
    n_cols = len(orders)
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 4))

    if n_cols == 1:
        axes = [axes]

    for ax, n in zip(axes, orders):
        r = recursive_step(x, n)
        c = classical_step(x, n)
        ax.plot(x, r, linewidth=2,
                label=f"Recursive  (deg {n + 1}, {n + 1} pieces)")
        ax.plot(x, c, "--", linewidth=2,
                label=f"Classical  (deg {2 * n + 1}, 1 piece)")
        ax.set_title(f"Order {n}  ($C^{n}$ smooth)")
        ax.set_xlabel("$x$")
        ax.set_ylabel("$f(x)$")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(0, 1)
        ax.set_ylim(-0.05, 1.05)

    fig.suptitle("Recursive vs Classical Smooth Step Functions", fontsize=13)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    print(f"Plot saved to {output_path}")
    plt.show()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Demonstrate smooth piecewise polynomial step functions.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--no-plot",
        action="store_true",
        help="Skip matplotlib plots (useful in environments without a display).",
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[1, 2, 3],
        metavar="N",
        help="Smoothness orders to demonstrate (default: 1 2 3).",
    )
    parser.add_argument(
        "--output",
        default="smooth_step_comparison.png",
        metavar="FILE",
        help="Output file for the comparison plot (default: smooth_step_comparison.png).",
    )
    args = parser.parse_args()

    print("Smooth Piecewise Polynomial Step Functions")
    print("=" * 45)
    print_comparison_table(orders=args.orders)

    if not args.no_plot:
        plot_comparison(orders=args.orders, output_path=args.output)


if __name__ == "__main__":
    main()
