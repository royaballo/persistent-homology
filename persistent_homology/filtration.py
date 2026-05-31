"""Vietoris-Rips filtration construction.

This module builds the Vietoris-Rips filtration of a finite point cloud:
the nested family of simplicial complexes obtained by connecting every set
of points whose pairwise distances all fall below a threshold.
"""

from itertools import combinations

import numpy as np
from scipy.spatial.distance import pdist  # type: ignore[import-untyped]
from scipy.spatial.distance import squareform


def vr_filtration(points, max_dim=2, max_eps=np.inf):
    """Construct a Vietoris-Rips filtration.

    Parameters
    ----------
    points : ndarray of shape (n, d)
        The input point cloud.
    max_dim : int
        Maximum simplex dimension to include.
    max_eps : float
        Maximum filtration value; simplices of larger diameter are dropped.

    Returns
    -------
    simplices : list of (filtration_value, simplex_tuple)
        Sorted by filtration value, then dimension, then lexicographically.
        Each simplex is a tuple of vertex indices.
    """
    n = len(points)
    dist_matrix = squareform(pdist(points))
    simplices = []

    # 0-simplices: all vertices at time 0
    for i in range(n):
        simplices.append((0.0, (i,)))

    # k-simplices for k = 1, ..., max_dim
    for k in range(1, max_dim + 1):
        for combo in combinations(range(n), k + 1):
            # Filtration value = max pairwise distance (the diameter)
            diam = max(dist_matrix[i][j]
                       for i, j in combinations(combo, 2))
            if diam <= max_eps:
                simplices.append((diam, combo))

    # Sort by filtration value, then dimension, then lex
    simplices.sort(key=lambda x: (x[0], len(x[1]), x[1]))
    return simplices
