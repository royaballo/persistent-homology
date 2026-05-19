# Persistent Homology

A pedagogical, from-scratch implementation of persistent homology for
Vietoris-Rips filtrations, developed as the computational component of an
MSc thesis.

<!-- TODO: one or two sentences describing the project and linking to the thesis PDF. -->

## Overview

<!-- TODO: describe what the code does:
     - builds the Vietoris-Rips filtration of a point cloud
     - computes persistent homology via three reduction strategies:
       standard, clearing, and cohomology (pCoh)
     - verified for correctness against Ripser and GUDHI
     Mention that the goal is clarity and a close correspondence with the
     theory, not production-grade performance. -->

## Installation

```bash
git clone https://github.com/<username>/persistent-homology.git
cd persistent-homology
pip install -e .
```

<!-- TODO: if you turn the project into an installable package, document
     `pip install -e .` here instead. -->

## Usage

<!-- TODO: a minimal, runnable example, e.g.:

```python
from persistent_homology.persistence import compute_persistence
from datasets.synthetic import make_circle

points = make_circle(n=30)
diagram = compute_persistence(points, max_dim=2, max_eps=2.0,
                              algorithm='cohomology')
```
-->

## Repository Structure

```
persistent_homology/   core implementation
    filtration.py          Vietoris-Rips complex construction
    boundary.py            boundary and coboundary matrices
    reduction.py           standard, clearing, and cohomology reductions
    persistence.py         end-to-end entry point
tests/                     correctness tests (pytest)
datasets/                  synthetic dataset generators
benchmarks/                performance benchmarking scripts
notebooks/                 verification and result notebooks
figures/                   generated figures (not tracked)
```

## Reproducing the Thesis Results

<!-- TODO: explain how to regenerate the tables and figures of the
     experimental chapter, e.g. which benchmark scripts to run and in
     what order. -->

## References

<!-- TODO: list the key papers, e.g.:
     - Zomorodian & Carlsson, Computing persistent homology (2005)
     - de Silva, Morozov & Vejdemo-Johansson, Dualities in persistent
       (co)homology (2011)
     - Bauer, Ripser (2021)
-->

## License

This project is released under the MIT License. See [LICENSE](LICENSE).
