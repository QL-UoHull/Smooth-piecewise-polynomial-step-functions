from __future__ import annotations

from dataclasses import dataclass
from math import atan, comb, cos, pi, tanh
from typing import Callable, Iterable


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def generalized_smoothstep(x: float, order: int = 2) -> float:
    if order < 0:
        raise ValueError("order must be non-negative")
    x = clamp01(x)
    series = 0.0
    for k in range(order + 1):
        series += (
            comb(order + k, k)
            * comb((2 * order) + 1, order - k)
            * ((-x) ** k)
        )
    return (x ** (order + 1)) * series


def linear_step(x: float) -> float:
    return clamp01(x)


def cubic_smoothstep(x: float) -> float:
    x = clamp01(x)
    return x * x * (3.0 - (2.0 * x))


def quintic_smootherstep(x: float) -> float:
    x = clamp01(x)
    return x * x * x * ((x * ((6.0 * x) - 15.0)) + 10.0)


def septic_smoothstep(x: float) -> float:
    x = clamp01(x)
    return (35.0 * (x ** 4)) - (84.0 * (x ** 5)) + (70.0 * (x ** 6)) - (20.0 * (x ** 7))


def cosine_ease(x: float) -> float:
    x = clamp01(x)
    return 0.5 - (0.5 * cos(pi * x))


def _normalize(value: float, start: float, end: float) -> float:
    return (value - start) / (end - start)


def logistic_sigmoid(x: float, steepness: float = 12.0) -> float:
    import math

    raw = 1.0 / (1.0 + math.exp(-steepness * (clamp01(x) - 0.5)))
    lo = 1.0 / (1.0 + math.exp(steepness / 2.0))
    hi = 1.0 / (1.0 + math.exp(-steepness / 2.0))
    return _normalize(raw, lo, hi)


def tanh_sigmoid(x: float, steepness: float = 3.0) -> float:
    raw = tanh(steepness * (clamp01(x) - 0.5))
    lo = tanh(-steepness / 2.0)
    hi = tanh(steepness / 2.0)
    return _normalize(raw, lo, hi)


def arctan_sigmoid(x: float, steepness: float = 8.0) -> float:
    raw = atan(steepness * (clamp01(x) - 0.5))
    lo = atan(-steepness / 2.0)
    hi = atan(steepness / 2.0)
    return _normalize(raw, lo, hi)


def midpoint_slope(function: Callable[[float], float], step: float = 1e-6) -> float:
    return (function(0.5 + step) - function(0.5 - step)) / (2.0 * step)


@dataclass(frozen=True)
class ComparisonFunction:
    name: str
    family: str
    function: Callable[[float], float]
    exact_endpoints: bool
    compact_support: bool
    endpoint_flatness_order: str
    notes: str


def default_comparison_functions(orders: Iterable[int] = (2, 3, 5)) -> list[ComparisonFunction]:
    functions = [
        ComparisonFunction(
            name=f"smooth_piecewise_polynomial(order={order})",
            family="this repository",
            function=lambda x, order=order: generalized_smoothstep(x, order),
            exact_endpoints=True,
            compact_support=True,
            endpoint_flatness_order=str(order),
            notes="Adjustable flatness order with exact compact support.",
        )
        for order in orders
    ]
    functions.extend(
        [
            ComparisonFunction(
                name="linear_step",
                family="common online baseline",
                function=linear_step,
                exact_endpoints=True,
                compact_support=True,
                endpoint_flatness_order="0",
                notes="Fast baseline, but no endpoint flattening.",
            ),
            ComparisonFunction(
                name="cubic_smoothstep",
                family="common online baseline",
                function=cubic_smoothstep,
                exact_endpoints=True,
                compact_support=True,
                endpoint_flatness_order="1",
                notes="Widely published cubic smoothstep.",
            ),
            ComparisonFunction(
                name="quintic_smootherstep",
                family="common online baseline",
                function=quintic_smootherstep,
                exact_endpoints=True,
                compact_support=True,
                endpoint_flatness_order="2",
                notes="Widely published quintic smootherstep.",
            ),
            ComparisonFunction(
                name="septic_smoothstep",
                family="common online baseline",
                function=septic_smoothstep,
                exact_endpoints=True,
                compact_support=True,
                endpoint_flatness_order="3",
                notes="A higher-order fixed polynomial often shared online.",
            ),
            ComparisonFunction(
                name="cosine_ease",
                family="common online baseline",
                function=cosine_ease,
                exact_endpoints=True,
                compact_support=True,
                endpoint_flatness_order="1",
                notes="Sinusoidal easing with zero first derivative only.",
            ),
            ComparisonFunction(
                name="logistic_sigmoid",
                family="common online baseline",
                function=logistic_sigmoid,
                exact_endpoints=True,
                compact_support=False,
                endpoint_flatness_order="0",
                notes="Smooth everywhere, but not compactly supported before normalization.",
            ),
            ComparisonFunction(
                name="tanh_sigmoid",
                family="common online baseline",
                function=tanh_sigmoid,
                exact_endpoints=True,
                compact_support=False,
                endpoint_flatness_order="0",
                notes="Analytic sigmoid with non-zero slope at the interval boundaries.",
            ),
            ComparisonFunction(
                name="arctan_sigmoid",
                family="common online baseline",
                function=arctan_sigmoid,
                exact_endpoints=True,
                compact_support=False,
                endpoint_flatness_order="0",
                notes="Another popular normalized sigmoid baseline.",
            ),
        ]
    )
    return functions
