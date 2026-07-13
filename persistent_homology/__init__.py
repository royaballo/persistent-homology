"""Persistent homology: a from-scratch implementation for Vietoris-Rips filtrations.

The package computes persistent homology of point clouds through three
interchangeable reduction strategies (standard, clearing, cohomology),
exposed through the single entry point :func:`compute_persistence`.

    >>> from persistent_homology import compute_persistence
    >>> from persistent_homology.datasets.synthetic import make_circle
    >>> dgm = compute_persistence(make_circle(n=30), max_dim=1, max_eps=2.0)

Invalid arguments raise ``ValueError`` or ``TypeError`` before any work is
done. A filtration large enough to risk exhausting memory raises a
:class:`FiltrationSizeWarning` rather than silently running for hours.
"""

from .boundary import build_boundary_matrix
from .filtration import vr_filtration
from .persistence import compute_persistence, extract_diagram
from .plotting import plot_persistence_diagram, plot_loglog_scaling
from .reduction import (add_columns_z2, reduce_pcoh, reduce_standard,
                        reduce_with_clearing)
from .validation import FiltrationSizeWarning

__version__ = "0.1.3"

__all__ = [
    # Main entry point
    "compute_persistence",
    # Pipeline stages
    "vr_filtration",
    "build_boundary_matrix",
    "reduce_standard",
    "reduce_with_clearing",
    "reduce_pcoh",
    "extract_diagram",
    "add_columns_z2",
    # Plotting
    "plot_persistence_diagram",
    "plot_loglog_scaling",
    # Warnings
    "FiltrationSizeWarning",
    "__version__",
]
