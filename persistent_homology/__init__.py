"""Persistent homology: a from-scratch implementation for Vietoris-Rips filtrations.

The package computes persistent homology of point clouds through three
interchangeable reduction strategies (standard, clearing, cohomology),
exposed through the single entry point :func:`compute_persistence`.
"""

from .boundary import build_boundary_matrix
from .filtration import vr_filtration
from .persistence import compute_persistence, extract_diagram
from .reduction import (add_columns_z2, reduce_pcoh, reduce_standard,
                        reduce_with_clearing)

__all__ = [
    "compute_persistence",
    "vr_filtration",
    "build_boundary_matrix",
    "add_columns_z2",
    "reduce_standard",
    "reduce_with_clearing",
    "reduce_pcoh",
    "extract_diagram",
]
