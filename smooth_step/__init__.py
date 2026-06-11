"""
smooth_step
===========

Python implementations of smooth piecewise polynomial step functions.

Two families are provided:

recursive
    Piecewise polynomial functions constructed via the B-spline integral
    approach. Order-n function uses degree-(n+1) polynomial pieces and is
    C^n smooth — roughly half the polynomial degree of the classical approach.

classical
    Single-polynomial constructions (cubic smoothstep, quintic smootherstep,
    and higher-order analogues) based on endpoint interpolation.

Quick usage::

    import numpy as np
    from smooth_step.recursive import smooth_step as recursive_step
    from smooth_step.classical import smooth_step as classical_step

    x = np.linspace(0, 1, 100)
    y_rec = recursive_step(x, order=2)   # C^2 smooth, piecewise cubic
    y_cls = classical_step(x, order=2)   # C^2 smooth, quintic polynomial
"""

from smooth_step.recursive import smooth_step as recursive_smooth_step
from smooth_step.classical import smooth_step as classical_smooth_step

__all__ = ["recursive_smooth_step", "classical_smooth_step"]
__version__ = "0.1.0"
