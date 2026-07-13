"""Input validation for the public API.
"""

import warnings
from math import comb

import numpy as np

# The three reduction strategies recognised by ``compute_persistence``.
VALID_ALGORITHMS = ("standard", "clearing", "cohomology")

# Above this many simplices, a Vietoris-Rips filtration is likely to
# exhaust memory on a typical machine. The threshold is deliberately
# generous: it is a guard rail, not a hard limit.
SIMPLEX_WARNING_THRESHOLD = 5_000_000


class FiltrationSizeWarning(UserWarning):
    """The requested filtration may be too large to compute.

    Emitted by :func:`check_filtration_size` when the number of simplices
    implied by ``n`` points at ``max_dim`` is large enough to risk
    exhausting memory. It is a warning rather than an error because the
    estimate ignores ``max_eps``, which can cut the true count by orders
    of magnitude.

    It can be silenced, or promoted to an error, in the usual way::

        import warnings
        from persistent_homology import FiltrationSizeWarning

        warnings.simplefilter("ignore", FiltrationSizeWarning)
        warnings.simplefilter("error", FiltrationSizeWarning)
    """


def validate_point_cloud(points):
    """Check and normalise a point cloud.

    Parameters
    ----------
    points : array_like
        Expected to be convertible to a 2D float array of shape
        ``(n_points, n_features)``.

    Returns
    -------
    points : ndarray of shape (n, d)
        The input as a contiguous float array.

    Raises
    ------
    TypeError
        If ``points`` cannot be converted to a numeric array.
    ValueError
        If ``points`` is not 2D, holds fewer than two points, or
        contains NaN or infinite values.
    """
    try:
        points = np.asarray(points, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            "points must be convertible to a numeric array of shape "
            f"(n_points, n_features); got {type(points).__name__}"
        ) from exc

    if points.ndim != 2:
        raise ValueError(
            "points must be a 2D array of shape (n_points, n_features); "
            f"got an array of shape {points.shape} ({points.ndim}D). "
            "A single point should be passed as [[x, y]], not [x, y]."
        )

    n, d = points.shape

    if n < 2:
        raise ValueError(
            f"a filtration needs at least 2 points; got {n}."
        )

    if d < 1:
        raise ValueError(
            f"points must have at least one feature; got shape {points.shape}."
        )

    if not np.isfinite(points).all():
        n_bad = int((~np.isfinite(points)).sum())
        raise ValueError(
            f"points contains {n_bad} NaN or infinite value(s). "
            "Pairwise distances are undefined for such points; remove or "
            "impute them before computing persistence."
        )

    return points


def validate_max_dim(max_dim):
    """Check the maximum homology dimension.

    Parameters
    ----------
    max_dim : int
        Must be a non-negative integer.

    Returns
    -------
    max_dim : int

    Raises
    ------
    TypeError
        If ``max_dim`` is not an integer.
    ValueError
        If ``max_dim`` is negative.
    """
    if isinstance(max_dim, bool) or not isinstance(max_dim, (int, np.integer)):
        raise TypeError(
            f"max_dim must be an integer; got {type(max_dim).__name__} "
            f"({max_dim!r})."
        )
    if max_dim < 0:
        raise ValueError(
            f"max_dim must be non-negative; got {max_dim}. "
            "Use max_dim=0 for connected components only."
        )
    return int(max_dim)


def validate_max_eps(max_eps):
    """Check the maximum filtration value.

    Parameters
    ----------
    max_eps : float
        Must be positive. ``numpy.inf`` is allowed and means "no cap".

    Returns
    -------
    max_eps : float

    Raises
    ------
    TypeError
        If ``max_eps`` is not a real number.
    ValueError
        If ``max_eps`` is zero, negative, or NaN.
    """
    if isinstance(max_eps, bool) or not isinstance(
        max_eps, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"max_eps must be a number; got {type(max_eps).__name__} "
            f"({max_eps!r})."
        )

    max_eps = float(max_eps)

    if np.isnan(max_eps):
        raise ValueError("max_eps must not be NaN.")
    if max_eps <= 0:
        raise ValueError(
            f"max_eps must be positive; got {max_eps}. "
            "Use numpy.inf (the default) to leave the filtration uncapped."
        )
    return max_eps


def validate_algorithm(algorithm):
    """Check the reduction strategy name.

    Parameters
    ----------
    algorithm : {'standard', 'clearing', 'cohomology'}

    Returns
    -------
    algorithm : str

    Raises
    ------
    ValueError
        If ``algorithm`` is not one of the three recognised names.
    """
    if algorithm not in VALID_ALGORITHMS:
        raise ValueError(
            f"unknown algorithm {algorithm!r}; expected one of "
            f"{list(VALID_ALGORITHMS)}."
        )
    return algorithm


def estimate_n_simplices(n_points, max_dim):
    """Upper bound on the number of simplices in a Vietoris-Rips complex.

    Counts every subset of size 1 to ``max_dim + 1``, i.e. the full
    skeleton with no distance cap. The true count is smaller whenever
    ``max_eps`` is finite, often by orders of magnitude, so this is a
    ceiling and not a prediction.

    Parameters
    ----------
    n_points : int
    max_dim : int

    Returns
    -------
    n_simplices : int
    """
    return sum(comb(n_points, k + 1) for k in range(max_dim + 1))


def check_filtration_size(n_points, max_dim, max_eps, threshold=None):
    """Warn if the requested filtration is likely to exhaust memory.

    A Vietoris-Rips filtration grows combinatorially: 500 points at
    ``max_dim=3`` admits over two billion simplices. Without a warning
    the caller sees only an unresponsive process, so this raises a
    :class:`FiltrationSizeWarning` before any work is done.

    The check is skipped when ``max_eps`` is finite, since a distance cap
    typically removes the vast majority of the simplices counted here and
    the bound would be alarmist.

    Parameters
    ----------
    n_points : int
    max_dim : int
    max_eps : float
        Only an infinite value triggers the check.
    threshold : int, optional
        Simplex count above which to warn. Defaults to
        :data:`SIMPLEX_WARNING_THRESHOLD`.

    Warns
    -----
    FiltrationSizeWarning
    """
    if threshold is None:
        threshold = SIMPLEX_WARNING_THRESHOLD

    if np.isfinite(max_eps):
        return

    n_simplices = estimate_n_simplices(n_points, max_dim)
    if n_simplices > threshold:
        warnings.warn(
            f"the filtration of {n_points} points up to dimension {max_dim} "
            f"may contain around {n_simplices:.2e} simplices, which is likely "
            "to exhaust memory. Set a finite max_eps, lower max_dim, or "
            "subsample the point cloud.",
            FiltrationSizeWarning,
            stacklevel=3,
        )
