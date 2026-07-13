"""End-to-end persistent homology computation.

This module ties the pipeline together: it builds the Vietoris-Rips
filtration of a point cloud, applies the requested reduction strategy,
and returns the resulting persistence diagram.
"""

import logging

import numpy as np

from .boundary import build_boundary_matrix
from .filtration import vr_filtration
from .reduction import reduce_pcoh, reduce_standard, reduce_with_clearing
from .validation import (validate_algorithm, validate_max_dim,
                         validate_max_eps, validate_point_cloud)


logger = logging.getLogger(__name__)


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

    Raises
    ------
    IndexError
        If a pair refers to a position outside the filtration, which
        means the pairs and the filtration do not belong together.
    """
    m = len(filtration)

    # Pairs and filtration must come from the same computation. Catching a
    # stray index here gives a diagnosable error instead of a bare
    # IndexError from deep inside the loop below.
    for (birth_idx, death_idx) in pairs:
        if not (0 <= birth_idx < m and 0 <= death_idx < m):
            raise IndexError(
                f"persistence pair ({birth_idx}, {death_idx}) falls outside "
                f"the filtration, which holds {m} simplices. The pairs and "
                "the filtration must come from the same computation."
            )

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
        The input point cloud. Must hold at least two points and contain
        no NaN or infinite values.
    max_dim : int
        Maximum homology dimension to compute. Must be non-negative.
    max_eps : float
        Maximum filtration value. Must be positive; ``numpy.inf`` (the
        default) leaves the filtration uncapped.
    algorithm : {'standard', 'clearing', 'cohomology'}
        The reduction strategy to use. All three return the same diagram;
        they differ only in cost.

    Returns
    -------
    diagram : dict
        Maps each dimension to a list of "(birth, death)" pairs. Essential
        features carry ``float('inf')`` as their death value.

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    ValueError
        If ``points`` is not a valid 2D point cloud, if ``max_dim`` is
        negative, if ``max_eps`` is not positive, or if ``algorithm`` is
        not one of the three recognised values.

    Warns
    -----
    FiltrationSizeWarning
        If the filtration is large enough to risk exhausting memory.

    Examples
    --------
    >>> from persistent_homology.datasets.synthetic import make_circle
    >>> pts = make_circle(n=30, noise=0.05)
    >>> dgm = compute_persistence(pts, max_dim=1, max_eps=2.0)
    >>> sorted(dgm)
    [0, 1]
    """
    # Validate everything before any work is done.
    points = validate_point_cloud(points)
    max_dim = validate_max_dim(max_dim)
    max_eps = validate_max_eps(max_eps)
    algorithm = validate_algorithm(algorithm)

    filt = vr_filtration(points, max_dim, max_eps)
    logger.info("Filtration built with %d simplices.", len(filt))

    if algorithm == 'cohomology':
        pairs = reduce_pcoh(filt)
    elif algorithm == 'clearing':
        cols = build_boundary_matrix(filt)
        _, pairs = reduce_with_clearing(cols, filt)
    else:  # 'standard'; the name was validated above
        cols = build_boundary_matrix(filt)
        _, pairs = reduce_standard(cols)

    return extract_diagram(pairs, filt)
