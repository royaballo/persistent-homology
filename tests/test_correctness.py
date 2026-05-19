"""Correctness tests for the persistent homology pipeline.

These tests check two things: that small hand-verifiable filtrations give
the expected Betti numbers, and that the three reduction strategies always
agree with one another.
"""

import numpy as np
import pytest  # type: ignore[import-not-found]

from persistent_homology import compute_persistence

ALGORITHMS = ["standard", "clearing", "cohomology"]


def _essential_count(diagram, dim):
    """Count essential (infinite-persistence) intervals in dimension 'dim'.

    The number of essential features in dimension k equals the k-th Betti
    number of the final complex.
    """
    intervals = diagram.get(dim, [])
    return sum(1 for _, death in intervals if death == float("inf"))


def _max_finite_persistence(diagram, dim):
    """Largest persistence among the finite intervals in dimension 'dim'."""
    intervals = diagram.get(dim, [])
    finite = [(b, d) for b, d in intervals if d != float("inf")]
    return max((d - b for b, d in finite), default=0.0)


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_filled_triangle(algorithm):
    """A filled triangle is contractible: one component, no loop."""
    triangle = np.array([[0.0, 0.0], [1.0, 0.0], [0.5, 0.87]])
    diagram = compute_persistence(triangle, max_dim=2, max_eps=5.0,
                                  algorithm=algorithm)
    # One essential H0 feature (the single connected component).
    assert _essential_count(diagram, 0) == 1
    # No essential H1 feature, and no persistent finite loop.
    assert _essential_count(diagram, 1) == 0
    assert _max_finite_persistence(diagram, 1) < 1e-6


@pytest.mark.parametrize("algorithm", ALGORITHMS)
def test_circle_has_one_loop(algorithm):
    """A sampled circle should exhibit a single dominant H1 feature."""
    rng = np.random.default_rng(0)
    theta = rng.uniform(0, 2 * np.pi, 20)
    circle = np.column_stack([np.cos(theta), np.sin(theta)])
    diagram = compute_persistence(circle, max_dim=2, max_eps=3.0,
                                  algorithm=algorithm)
    # The longest-lived H1 interval is the loop of the circle.
    assert _max_finite_persistence(diagram, 1) > 0.5


def test_all_algorithms_agree():
    """The three reduction strategies must produce identical diagrams."""
    rng = np.random.default_rng(1)
    points = rng.uniform(0, 1, (15, 3))

    diagrams = {
        algo: compute_persistence(points, max_dim=2, max_eps=2.0,
                                  algorithm=algo)
        for algo in ALGORITHMS
    }

    reference = diagrams["standard"]
    for algo in ("clearing", "cohomology"):
        for dim in reference:
            ref_sorted = sorted(reference[dim])
            got_sorted = sorted(diagrams[algo].get(dim, []))
            assert ref_sorted == got_sorted, (
                f"{algo} disagrees with standard in dimension {dim}"
            )


def test_unknown_algorithm_raises():
    """An unrecognised algorithm name should raise ValueError."""
    points = np.array([[0.0, 0.0], [1.0, 0.0]])
    with pytest.raises(ValueError):
        compute_persistence(points, algorithm="not-an-algorithm")
