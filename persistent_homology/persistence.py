"""End-to-end persistent homology computation.

This module ties the pipeline together: it builds the Vietoris-Rips
filtration of a point cloud, applies the requested reduction strategy,
and returns the resulting persistence diagram.
"""

import numpy as np

from .boundary import build_boundary_matrix
from .filtration import vr_filtration
from .reduction import reduce_pcoh, reduce_standard, reduce_with_clearing


def extract_diagram(pairs, filtration):
    """Convert index pairs into a persistence diagram.

    Essential features are births with no matching death; they correspond
    to intervals "[b, infinity)".

    Parameters
    ----------
    pairs : list of (birth_idx, death_idx)
        Persistence pairs as filtration indices.
    filtration : list of (filtration_value, simplex_tuple)
        The filtration the pairs index into.

    Returns
    -------
    diagram : dict
        Maps each dimension to a list of "(birth_value, death_value)"
        pairs. Infinite intervals use "float('inf')" as the death value.
    """
    diagram = {}

    # Indices that appear as births or deaths in some pair
    paired_births = {b for (b, _) in pairs}
    paired_deaths = {d for (_, d) in pairs}

    # Finite intervals: one per persistence pair
    for (birth_idx, death_idx) in pairs:
        birth_val = filtration[birth_idx][0]
        death_val = filtration[death_idx][0]
        dim = len(filtration[birth_idx][1]) - 1
        diagram.setdefault(dim, []).append(
            (birth_val, death_val))

    # Infinite intervals: births with no matching death.
    for idx, (val, simplex) in enumerate(filtration):
        is_unpaired_birth = (idx not in paired_births
                             and idx not in paired_deaths)
        if is_unpaired_birth:
            dim = len(simplex) - 1
            diagram.setdefault(dim, []).append(
                (val, float('inf')))

    return diagram


def compute_persistence(points, max_dim=2, max_eps=np.inf,
                        algorithm='clearing'):
    """Compute the persistent homology of a point cloud.

    Parameters
    ----------
    points : ndarray of shape (n, d)
        The input point cloud.
    max_dim : int
        Maximum homology dimension to compute.
    max_eps : float
        Maximum filtration value.
    algorithm : {'standard', 'clearing', 'cohomology'}
        The reduction strategy to use.

    Returns
    -------
    diagram : dict
        Maps each dimension to a list of "(birth, death)" pairs.

    Raises
    ------
    ValueError
        If "algorithm" is not one of the three recognised values.
    """
    filt = vr_filtration(points, max_dim, max_eps)
    print(f"Filtration built with {len(filt)} simplices.")

    if algorithm == 'cohomology':
        pairs = reduce_pcoh(filt)
    elif algorithm == 'clearing':
        cols = build_boundary_matrix(filt)
        _, pairs = reduce_with_clearing(cols, filt)
    elif algorithm == 'standard':
        cols = build_boundary_matrix(filt)
        _, pairs = reduce_standard(cols)
    else:
        raise ValueError(f"Unknown algorithm: {algorithm!r}")

    return extract_diagram(pairs, filt)
