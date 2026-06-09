from __future__ import annotations

import argparse
from typing import Iterable

from .core import ComparisonFunction, default_comparison_functions, midpoint_slope


def comparison_rows(orders: Iterable[int] = (2, 3, 5)) -> list[dict[str, str]]:
    rows = []
    for item in default_comparison_functions(orders):
        rows.append(
            {
                "name": item.name,
                "family": item.family,
                "exact_endpoints": "yes" if item.exact_endpoints else "no",
                "compact_support": "yes" if item.compact_support else "no",
                "endpoint_flatness": item.endpoint_flatness_order,
                "midpoint_slope": f"{midpoint_slope(item.function):.3f}",
                "notes": item.notes,
            }
        )
    return rows


def _format_table(rows: list[dict[str, str]], columns: list[tuple[str, str]]) -> str:
    widths = {
        key: max(len(title), *(len(row[key]) for row in rows))
        for key, title in columns
    }
    header = " | ".join(title.ljust(widths[key]) for key, title in columns)
    divider = "-+-".join("-" * widths[key] for key, _ in columns)
    body = [
        " | ".join(row[key].ljust(widths[key]) for key, _ in columns)
        for row in rows
    ]
    return "\n".join([header, divider, *body])


def _takeaways(functions: list[ComparisonFunction]) -> list[str]:
    repo_orders = [item.endpoint_flatness_order for item in functions if item.family == "this repository"]
    max_order = max(int(order) for order in repo_orders)
    return [
        f"The smooth piecewise polynomial family scales to any chosen endpoint flatness order; this report includes order {max_order}.",
        "Popular cubic and quintic online formulas appear as low-order special cases of the same polynomial family.",
        "Normalized sigmoid-style alternatives stay smooth in the middle but give up compact polynomial support and endpoint flatness.",
    ]


def render_report(orders: Iterable[int] = (2, 3, 5)) -> str:
    functions = default_comparison_functions(orders)
    rows = comparison_rows(orders)
    columns = [
        ("name", "function"),
        ("family", "source"),
        ("exact_endpoints", "exact 0/1"),
        ("compact_support", "compact"),
        ("endpoint_flatness", "flatness order"),
        ("midpoint_slope", "mid slope"),
        ("notes", "notes"),
    ]
    takeaways = "\n".join(f"- {line}" for line in _takeaways(functions))
    return (
        "Smooth step comparison report\n"
        "=============================\n\n"
        + _format_table(rows, columns)
        + "\n\nKey takeaways\n-------------\n"
        + takeaways
        + "\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Compare smooth piecewise polynomial step functions with common online alternatives."
    )
    parser.add_argument(
        "--orders",
        type=int,
        nargs="+",
        default=[2, 3, 5],
        help="Orders of the repository's smooth piecewise polynomial family to include.",
    )
    args = parser.parse_args(argv)
    print(render_report(args.orders))
    return 0
