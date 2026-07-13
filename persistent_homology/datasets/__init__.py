"""Dataset generators and loaders for the persistent homology experiments.

The synthetic samplers have known Betti numbers and are used for both
correctness verification and benchmarking. The protein loaders require
Biopython, an optional dependency; they are not imported here, so this
module stays importable without it.

    >>> from persistent_homology.datasets import make_circle
    >>> pts = make_circle(n=30, noise=0.05)
"""

from .synthetic import make_circle, make_cube, make_sphere, make_torus

__all__ = ["make_circle", "make_sphere", "make_torus", "make_cube"]
