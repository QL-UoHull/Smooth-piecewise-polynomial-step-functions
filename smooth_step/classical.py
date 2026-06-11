"""
Classical polynomial smooth step functions for comparison.

Each function is a **single** polynomial of degree :math:`2n+1` that
achieves :math:`C^n` smoothness with endpoint values 0 and 1 and
vanishing derivatives up to order *n* at both endpoints.

**Construction**

The order-*n* classical smooth step polynomial is given by the Bernstein
representation:

.. math::

    S(x) = x^{n+1} \\sum_{k=0}^{n} \\binom{n+k}{k} (1 - x)^k.

This coincides with the regularised incomplete Beta function
:math:`I_x(n+1,\\, n+1)`.

Special cases:

* **Order 1** (cubic smoothstep): :math:`3x^2 - 2x^3`
* **Order 2** (quintic smootherstep): :math:`10x^3 - 15x^4 + 6x^5`
* **Order 3**: :math:`35x^4 - 84x^5 + 70x^6 - 20x^7`

**Limitation**

Each order requires solving a new interpolation problem from scratch.
There is no recursive relation connecting the order-*n* function to
the order-:math:`(n-1)` function, and the polynomial degree grows as
:math:`2n+1` (degree 3, 5, 7, …).
"""

from math import comb

import numpy as np


def smooth_step(x, order):
    """Evaluate the classical degree-``(2*order+1)`` smooth step polynomial.

    Uses the Bernstein representation:

    .. math::

        S(x) = x^{n+1} \\sum_{k=0}^{n} \\binom{n+k}{k} (1-x)^k, \\quad n = \\text{order}.

    Parameters
    ----------
    x : array-like
        Input values.  Typically in ``[0, 1]``; values outside are clamped.
    order : int
        Smoothness order *n* ≥ 1.  The result is a polynomial of degree
        ``2*order + 1``.

    Returns
    -------
    numpy.ndarray
        Function values in ``[0, 1]``.

    Raises
    ------
    ValueError
        If *order* is less than 1.

    Examples
    --------
    >>> import numpy as np
    >>> from smooth_step.classical import smooth_step
    >>> x = np.array([0.0, 0.5, 1.0])
    >>> smooth_step(x, order=1)  # cubic smoothstep: 3x^2 - 2x^3
    array([0. , 0.5, 1. ])
    >>> smooth_step(x, order=2)  # quintic smootherstep: 10x^3 - 15x^4 + 6x^5
    array([0. , 0.5, 1. ])
    """
    if isinstance(order, bool) or not (
        isinstance(order, (int, np.integer))
        or (isinstance(order, float) and order.is_integer())
    ):
        raise ValueError(f"order must be a positive integer, got {order!r}")
    n = int(order)
    if n < 1:
        raise ValueError(f"order must be a positive integer, got {order!r}")

    x = np.clip(np.asarray(x, dtype=float), 0.0, 1.0)

    # S(x) = x^(n+1) * sum_{k=0}^{n} C(n+k, k) * (1-x)^k
    one_minus_x = 1.0 - x
    poly_sum = np.zeros_like(x)
    for k in range(n + 1):
        poly_sum += comb(n + k, k) * one_minus_x ** k
    result = x ** (n + 1) * poly_sum

    return np.clip(result, 0.0, 1.0)
