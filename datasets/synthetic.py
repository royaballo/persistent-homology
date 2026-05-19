"""Synthetic dataset generators with known topology.

Each generator returns a point cloud sampled from a space whose Betti
numbers are known in advance, which makes them suitable both for
correctness verification and for performance benchmarking. All generators
accept a "seed" argument so that every dataset is fully reproducible.

Expected topology:

* :func:'make_circle' -- circle S^1: beta_0 = 1, beta_1 = 1.
* :func:'make_sphere' -- sphere S^2: beta_0 = 1, beta_1 = 0, beta_2 = 1.
* :func:'make_torus'  -- torus T^2:  beta_0 = 1, beta_1 = 2, beta_2 = 1.
* :func:'make_cube'   -- random cube: contractible, beta_0 = 1, rest 0.
"""

import numpy as np


def make_circle(n=30, sigma=0.05, seed=0):
    """Sample a noisy unit circle in R^2.

    Parameters
    ----------
    n : int
        Number of points.
    sigma : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    points : ndarray of shape (n, 2)
    """
    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    return pts + sigma * rng.standard_normal((n, 2))


def make_sphere(n=100, sigma=0.02, seed=0):
    """Sample the unit sphere S^2 using a Fibonacci spiral.

    The Fibonacci spiral spreads points almost uniformly over the sphere;
    a small Gaussian perturbation is added.

    Parameters
    ----------
    n : int
        Number of points.
    sigma : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Random seed for the perturbation.

    Returns
    -------
    points : ndarray of shape (n, 3)
    """
    indices = np.arange(n, dtype=float) + 0.5
    phi = np.arccos(1 - 2 * indices / n)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    pts = np.column_stack([
        np.sin(phi) * np.cos(theta),
        np.sin(phi) * np.sin(theta),
        np.cos(phi),
    ])
    rng = np.random.default_rng(seed)
    return pts + sigma * rng.standard_normal(pts.shape)


def make_torus(n_u=20, n_v=5, R=3.0, r=1.0, sigma=0.02, seed=0):
    """Sample the torus T^2 on a regular grid.

    Points are placed on a regular "n_u" by "n_v" grid in the two
    angular coordinates, giving "n_u * n_v" points in total.

    Parameters
    ----------
    n_u : int
        Number of samples around the major (tube) circle.
    n_v : int
        Number of samples around the minor (cross-section) circle.
    R : float
        Major radius.
    r : float
        Minor radius.
    sigma : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Random seed for the perturbation.

    Returns
    -------
    points : ndarray of shape (n_u * n_v, 3)
    """
    u_vals = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v_vals = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    U, V = np.meshgrid(u_vals, v_vals)
    pts = np.column_stack([
        ((R + r * np.cos(V)) * np.cos(U)).flatten(),
        ((R + r * np.cos(V)) * np.sin(U)).flatten(),
        (r * np.sin(V)).flatten(),
    ])
    rng = np.random.default_rng(seed)
    return pts + sigma * rng.standard_normal(pts.shape)


def make_cube(n=100, seed=0):
    """Sample points uniformly from the unit cube [0, 1]^3.

    The unit cube is contractible, so no persistent topology is expected:
    this dataset serves as a negative control.

    Parameters
    ----------
    n : int
        Number of points.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    points : ndarray of shape (n, 3)
    """
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n, 3))
