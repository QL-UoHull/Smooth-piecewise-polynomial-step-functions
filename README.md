# Smooth Piecewise Polynomial Step Functions

Python implementation of **recursive piecewise polynomial smooth step functions**, with comparison against classical polynomial constructions (cubic smoothstep, quintic smootherstep, and higher-order variants).

## Overview

A smooth step function transitions from 0 to 1 over an interval while remaining smooth at the boundaries — appearing flat (zero derivatives) at each end. These functions appear in computer graphics, geometric modelling, numerical analysis, signal processing, and control theory.

The classical approach constructs a **single polynomial** of degree $2n+1$ that satisfies endpoint interpolation conditions. This work demonstrates an alternative: a **recursive piecewise polynomial** construction that achieves the same smoothness order using polynomials of roughly **half the degree**, with a natural recursive structure across orders.

| Smoothness | Classical | Recursive |
|:---:|:---:|:---:|
| $C^1$ | degree 3, 1 piece | degree 2, 2 pieces |
| $C^2$ | degree 5, 1 piece | degree 3, 3 pieces |
| $C^3$ | degree 7, 1 piece | degree 4, 4 pieces |
| $C^n$ | degree $2n+1$, 1 piece | degree $n+1$, $n+1$ pieces |

See [`NOTES.md`](NOTES.md) for a detailed mathematical discussion of why this matters.

## Installation

Requires Python 3.8+ and NumPy. Matplotlib is optional (needed for plots).

```bash
# Clone the repository
git clone https://github.com/QL-UoHull/Smooth-piecewise-polynomial-step-functions.git
cd Smooth-piecewise-polynomial-step-functions

# Install dependencies
pip install -r requirements.txt
```

## Usage

### As a library

```python
import numpy as np
from smooth_step.recursive import smooth_step as recursive_step
from smooth_step.classical import smooth_step as classical_step

x = np.linspace(0, 1, 100)

# Recursive C^2 smooth function: piecewise cubic, 3 pieces, degree 3
y_rec = recursive_step(x, order=2)

# Classical C^2 smooth function: quintic polynomial, degree 5
y_cls = classical_step(x, order=2)
```

### Demo script

Run the included demo to print a comparison table and (optionally) save a plot:

```bash
python demo.py                        # table + plot (requires matplotlib)
python demo.py --no-plot              # table only
python demo.py --orders 1 2 3 4       # choose smoothness orders
python demo.py --output my_plot.png   # custom output filename
```

Example output:

```
Order 2  (C^2 smooth)
  Recursive : degree-3 piecewise polynomials, 3 pieces
  Classical : single degree-5 polynomial

         x     Recursive     Classical    Difference
  --------  ------------  ------------  ------------
    0.0000      0.000000      0.000000     +0.000000
    0.2500      0.070312      0.103516     +0.033203
    0.5000      0.500000      0.500000     +0.000000
    0.7500      0.929688      0.896484     -0.033203
    1.0000      1.000000      1.000000     +0.000000
```

## Repository structure

```
.
├── smooth_step/
│   ├── __init__.py          # Package entry points
│   ├── recursive.py         # Recursive piecewise polynomial construction
│   └── classical.py         # Classical single-polynomial construction
├── tests/
│   └── test_smooth_step.py  # Unit tests
├── demo.py                  # Demonstration and comparison script
├── NOTES.md                 # Detailed mathematical background
├── requirements.txt
└── pyproject.toml
```

## Running the tests

```bash
python -m unittest discover -s tests
```

All tests verify mathematical properties: boundary values, symmetry, monotonicity, explicit polynomial formulas, and derivative conditions at the endpoints.

## Mathematical background

The recursive construction is based on integrating the uniform B-spline density. The order-$n$ smooth step function is:

$$T_n(x) = \frac{1}{(n+1)!} \sum_{j=0}^{n+1} (-1)^j \binom{n+1}{j} \bigl((n+1)x - j\bigr)_+^{\,n+1}$$

where $(u)_+ = \max(u, 0)$.

This function is $C^n$ smooth, has $n+1$ polynomial pieces each of degree $n+1$, and satisfies $T_n(0)=0$, $T_n(1)=1$, and $T_n^{(k)}(0)=T_n^{(k)}(1)=0$ for $k=1,\ldots,n$.

For full motivation and analysis — including why the recursive construction is mathematically preferable to the classical approach — see [`NOTES.md`](NOTES.md).

## Contributing

Contributions are welcome. Please open an issue or pull request. When adding new functionality, include corresponding tests and update the demo if appropriate.

