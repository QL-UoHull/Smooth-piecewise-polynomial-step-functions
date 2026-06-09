# Smooth-piecewise-polynomial-step-functions

Python code to demonstrate the strengths of smooth piecewise polynomial step
functions, with side-by-side comparisons against several commonly published
online alternatives.

The repository now includes:

- a reusable Python implementation of a generalized smooth piecewise polynomial
  step family with tunable endpoint flatness order
- common comparison baselines such as linear, cubic smoothstep, quintic
  smootherstep, cosine easing, logistic, tanh, and arctan variants
- a small CLI report that highlights compact support, exact endpoints, endpoint
  flatness order, and midpoint slope

## Usage

Run the comparison report from the repository root:

```bash
python -m smooth_steps --orders 2 3 5
```

Run the focused tests:

```bash
python -m unittest discover -s tests
```
