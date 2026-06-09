# Smooth Piecewise Polynomial Step Functions

This repository contains a Python implementation of the recursive smooth step construction provided by the repository owner, together with an interactive plotting demo.

## Core idea

The implementation is based on the recursive definition:

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

This formulation produces a family of piecewise-polynomial step functions with adjustable smoothness order `n`.

## Files

- `smoothstep_recursive.py` — Python implementation of `H` and `FF`
- `interactive_plot.py` — interactive Plotly demo showing the effect of varying `n`
- `requirements.txt` — dependencies for the demo

## Install

```bash
python -m pip install -r requirements.txt
```

## Run the interactive demo

```bash
python interactive_plot.py
```

This writes an HTML file `smoothstep_interactive.html` that you can open in a browser.
