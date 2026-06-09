# Smooth Piecewise Polynomial Step Functions

A research-oriented repository for recursive smooth piecewise polynomial step functions, including theory, reference implementations, examples, and comparative notes.

## Repository goals

- document the recursive formulation of the `H(s, n)` family,
- provide clean reference implementations in Python and other languages,
- demonstrate the effect of increasing smoothness order `n`,
- compare this family with standard smoothstep formulations often presented using higher-degree polynomials,
- collect notes, plots, and supplementary materials connected to the published work.

## Suggested repository structure

- `src/` — source implementations
- `examples/` — runnable demos and plotting scripts
- `notebooks/` — exploratory notebooks and visual comparisons
- `docs/` — derivations, notes, figures, and references
- `tests/` — verification tests for monotonicity, symmetry, and endpoint behavior
- `assets/` — images used by the documentation

## Current status

This repository is being organized as a professional research/code companion for the smooth step construction.

## Reference MATLAB definition

```matlab
function y=H(s, n)
if n==0
    y=FF(s, 0);
else
    y=FF(n*(s+1)/2, n);
end;

function y=FF(s, n)
if n==0
    y=(s==0)/2 +(s>0);
    return;
else
    t=s/n;
    y=t.*FF(s, n-1)+(1-t).*FF(s-1, n-1);
end;
```

## Planned additions

- Python implementation of `H` and `FF`
- interactive plotting demo
- comparison with standard smoothstep families
- derivation notes extracted from the paper and supporting materials

## Citation

If you use this repository in academic work, please cite the associated paper and link to this repository.
