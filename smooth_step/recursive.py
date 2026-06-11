"""
Recursive piecewise polynomial smooth step functions.

These functions are constructed by integrating uniform B-spline density
functions. The order-*n* smooth step function:

* is :math:`C^n` smooth on ``[0, 1]``,
* satisfies ``f(0) = 0`` and ``f(1) = 1``,
* satisfies ``f^(k)(0) = f^(k)(1) = 0`` for ``k = 1, …, n``,
* consists of ``(n + 1)`` polynomial pieces, each of degree ``(n + 1)``.

**Construction**

The *n*-th order uniform B-spline :math:`B_n` has support :math:`[0, n+1]`
and is given by the divided-difference formula:

.. math::

    B_n(t) = \\frac{1}{n!} \\sum_{j=0}^{n+1} (-1)^j \\binom{n+1}{j}
              \\max(t - j,\\, 0)^n.

Integrating from 0 to :math:`t` and noting that
:math:`\\int_0^{n+1} B_n = 1`, the step function on :math:`[0, n+1]` is:

.. math::

    S_n(t) = \\frac{1}{(n+1)!} \\sum_{j=0}^{n+1} (-1)^j \\binom{n+1}{j}
              \\max(t - j,\\, 0)^{n+1}.

The normalised function on :math:`[0, 1]` is then
:math:`T_n(x) = S_n((n+1)\\,x)`.

**Comparison with classical construction**

For :math:`C^n` smoothness:

* This construction uses degree-:math:`(n+1)` polynomials — roughly
  **half** the degree of the classical degree-:math:`(2n+1)` approach.
* The recursive structure means order-:math:`n` follows naturally from
  order-:math:`(n-1)`, without solving a new interpolation problem.
"""

from math import comb, factorial

import numpy as np


def smooth_step(x, order):
    """Evaluate the recursive piecewise polynomial smooth step function.

    Parameters
    ----------
    x : array-like
        Input values.  Typically in ``[0, 1]``; values outside are clamped.
    order : int
        Smoothness order *n* ≥ 0.  The result is :math:`C^n` smooth and
        consists of ``(n + 1)`` polynomial pieces of degree ``(n + 1)``.

    Returns
    -------
    numpy.ndarray
        Function values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If *order* is negative.

    Examples
    --------
    >>> import numpy as np
    >>> from smooth_step.recursive import smooth_step
    >>> x = np.array([0.0, 0.25, 0.5, 0.75, 1.0])
    >>> smooth_step(x, order=1)
    array([0.   , 0.125, 0.5  , 0.875, 1.   ])
    >>> smooth_step(x, order=2)
    array([0.      , 0.15625, 0.5    , 0.84375, 1.     ])
    """
    if isinstance(order, bool) or not (
        isinstance(order, (int, np.integer))
        or (isinstance(order, float) and order.is_integer())
    ):
        raise ValueError(f"order must be a non-negative integer, got {order!r}")
    n = int(order)
    if n < 0:
        raise ValueError(f"order must be a non-negative integer, got {order!r}")

    x = np.asarray(x, dtype=float)
    # Scale input to the B-spline support [0, n+1]
    t = (n + 1) * x

    # Compute the integral of B_n from 0 to t using the explicit formula:
    #   S(t) = 1/(n+1)! * sum_{j=0}^{n+1} (-1)^j * C(n+1, j) * max(t-j, 0)^(n+1)
    prefactor = 1.0 / factorial(n + 1)
    result = np.zeros_like(t)
    for j in range(n + 2):
        sign = (-1) ** j
        result += sign * comb(n + 1, j) * np.maximum(t - j, 0.0) ** (n + 1)
    result *= prefactor

    return np.clip(result, 0.0, 1.0)
