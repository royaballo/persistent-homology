# persistent-homology

[![PyPI version](https://img.shields.io/pypi/v/persistent-homology)](https://pypi.org/project/persistent-homology/)
[![Python versions](https://img.shields.io/pypi/pyversions/persistent-homology)](https://pypi.org/project/persistent-homology/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow)](https://github.com/royaballo/persistent-homology/blob/main/LICENSE)

A from-scratch, readable implementation of persistent homology for Vietoris-Rips filtrations.

This is not a speed-first library: Ripser and GUDHI already are, and they are hard to read. This one is the opposite. Every function maps onto a definition or an algorithm from the literature, so you can follow the path from a point cloud to a persistence diagram without taking any step on faith. It is meant to be read as much as run.

## Install

```bash
pip install persistent-homology
```

Requires Python 3.10+. The core install pulls in NumPy, SciPy, Matplotlib, and pandas.

## Quickstart

```python
from persistent_homology import compute_persistence, plot_persistence_diagram
from persistent_homology.datasets import make_circle

# 30 points on a noisy circle
points = make_circle(n=30, noise=0.05)

diagram = compute_persistence(
    points,
    max_dim=1,
    max_eps=2.0,
    algorithm="cohomology",   # or "standard", "clearing"
)

for dim, dgm in diagram.items():
    print(f"H{dim}: {len(dgm)} features")

plot_persistence_diagram(diagram, max_dim=1)
```

The circle gives one point sitting well above the diagonal in H1: the hole. Everything else is short-lived noise near the diagonal. That gap between signal and noise is what persistent homology is for.

Diagrams come back as `{dimension: [(birth, death), ...]}`. Essential features carry `float("inf")` as their death value and are drawn as triangles along a dashed infinity line.

## The three reduction strategies

All three return the same diagram. That equality is the content of the duality between persistent homology and cohomology, and the test suite checks it on every dataset. They differ only in cost.

| `algorithm` | What it is | Where it shines |
|---|---|---|
| `"standard"` | Plain left-to-right column reduction (Zomorodian-Carlsson) | The baseline; easiest to read |
| `"clearing"` | Standard reduction, but births in dimension *k* are zeroed once dimension *k+1* is done | Skips wasted work in high dimensions |
| `"cohomology"` | Reduces the anti-transpose instead (pCoh, de Silva-Morozov-Vejdemo-Johansson) | Much shorter columns on VR complexes; usually fastest |

## A note on size

Vietoris-Rips filtrations grow combinatorially: 300 points at `max_dim=3` admits roughly 3 x 10^8 simplices, which will exhaust memory on most machines. Rather than let that run silently, the package warns you before it starts:

```python
import warnings
from persistent_homology import FiltrationSizeWarning

warnings.simplefilter("error", FiltrationSizeWarning)  # or "ignore"
```

Setting a finite `max_eps` is the cheapest way to stay out of trouble, and it is what you almost always want anyway.

## Optional extras

```bash
pip install "persistent-homology[benchmark]"   # ripser + gudhi, for the correctness tests
pip install "persistent-homology[bio]"         # biopython, for loading PDB structures
```

## Citing

If you use this package in your research, please cite it:

```bibtex
@misc{aballo2026persistenthomology,
  author       = {Aballo, Roy and Gaba, Ya{\'e} Ulrich},
  title        = {Persistent Homology: A From-Scratch Implementation of
                  Vietoris-Rips Persistence in Python},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/royaballo/persistent-homology}},
  note         = {AIMS Senegal Master's Thesis}
}
```

The full citation will be updated once the thesis is published.

## More

This package is the computational half of an MSc thesis in Topological Data Analysis. The full repository holds the correctness tests against Ripser and GUDHI, the scaling benchmarks, a protein case study on haemoglobin, lysozyme, and GFP, and the notebooks that regenerate every figure.

[github.com/royaballo/persistent-homology](https://github.com/royaballo/persistent-homology)

## References

- Edelsbrunner, Letscher, Zomorodian. *Topological persistence and simplification* (2002).
- Zomorodian, Carlsson. *Computing persistent homology* (2005).
- de Silva, Morozov, Vejdemo-Johansson. *Dualities in persistent (co)homology* (2011).

## License

MIT. See [LICENSE](https://github.com/royaballo/persistent-homology/blob/main/LICENSE).
