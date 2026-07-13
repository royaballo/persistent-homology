"""Vietoris-Rips filtration construction.

This module builds the Vietoris-Rips filtration of a finite point cloud:
the nested family of simplicial complexes obtained by connecting every set
of points whose pairwise distances all fall below a threshold.
"""

from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform

from .validation import (check_filtration_size, validate_max_dim,
                         validate_max_eps, validate_point_cloud)


def vr_filtration(points, max_dim=2, max_eps=np.inf):
    """Construct a Vietoris-Rips filtration.

    Parameters
    ----------
    points : ndarray of shape (n, d)
        The input point cloud. Must hold at least two points and contain
        no NaN or infinite values.
    max_dim : int
        Maximum simplex dimension to include. Must be non-negative.
    max_eps : float
        Maximum filtration value; simplices of larger diameter are
        dropped. Must be positive; ``numpy.inf`` leaves the filtration
        uncapped.

    Returns
    -------
    simplices : list of (filtration_value, simplex_tuple)
        Sorted by filtration value, then dimension, then lexicographically.
        Each simplex is a tuple of vertex indices.

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    ValueError
        If ``points`` is not a valid 2D point cloud, if ``max_dim`` is
        negative, or if ``max_eps`` is not positive.

    Warns
    -----
    FiltrationSizeWarning
        If the filtration is large enough to risk exhausting memory.

    Examples
    --------
    >>> import numpy as np
    >>> pts = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    >>> filt = vr_filtration(pts, max_dim=1, max_eps=1.5)
    >>> filt[0]
    (0.0, (0,))
    """
    points = validate_point_cloud(points)
    max_dim = validate_max_dim(max_dim)
    max_eps = validate_max_eps(max_eps)

    n = len(points)
    check_filtration_size(n, max_dim, max_eps)

    # A k-simplex needs k+1 vertices, so dimensions beyond n-1 are empty.
    # Capping here keeps the loop below from iterating over combination
    # sets that are known to be empty.
    effective_max_dim = min(max_dim, n - 1)

    dist_matrix = squareform(pdist(points))
    simplices = []

    # 0-simplices: all vertices at time 0
    for i in range(n):
        simplices.append((0.0, (i,)))

    # k-simplices for k = 1, ..., effective_max_dim
    for k in range(1, effective_max_dim + 1):
        for combo in combinations(range(n), k + 1):
            # Filtration value = max pairwise distance (the diameter)
            diam = max(dist_matrix[i][j]
                       for i, j in combinations(combo, 2))
            if diam <= max_eps:
                simplices.append((diam, combo))

    # Sort by filtration value, then dimension, then lex
    simplices.sort(key=lambda x: (x[0], len(x[1]), x[1]))
    return simplices
