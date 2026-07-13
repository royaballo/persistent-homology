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


def _validate_n(n, name="n", minimum=2):
    """Check a sample count."""
    if isinstance(n, bool) or not isinstance(n, (int, np.integer)):
        raise TypeError(
            f"{name} must be an integer; got {type(n).__name__} ({n!r})."
        )
    if n < minimum:
        raise ValueError(
            f"{name} must be at least {minimum}; got {n}. A filtration needs "
            "at least two points."
        )
    return int(n)


def _validate_noise(noise):
    """Check a noise standard deviation."""
    if isinstance(noise, bool) or not isinstance(
        noise, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"noise must be a number; got {type(noise).__name__} ({noise!r})."
        )
    noise = float(noise)
    if not np.isfinite(noise):
        raise ValueError(f"noise must be finite; got {noise}.")
    if noise < 0:
        raise ValueError(
            f"noise is a standard deviation and must be non-negative; got "
            f"{noise}. Use noise=0 for an exact sample."
        )
    return noise


def _validate_radius(value, name):
    """Check a torus radius."""
    if isinstance(value, bool) or not isinstance(
        value, (int, float, np.integer, np.floating)
    ):
        raise TypeError(
            f"{name} must be a number; got {type(value).__name__} ({value!r})."
        )
    value = float(value)
    if not np.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a positive finite number; got {value}.")
    return value


def make_circle(n=30, noise=0.05, seed=0):
    """Sample a noisy unit circle in R^2.

    Parameters
    ----------
    n : int
        Number of points.
    noise : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    points : ndarray of shape (n, 2)

    Raises
    ------
    TypeError
        If "n" is not an integer, or "noise" is not a number.
    ValueError
        If "n" is below 2, or "noise" is negative.
    """
    n = _validate_n(n)
    noise = _validate_noise(noise)

    rng = np.random.default_rng(seed)
    theta = rng.uniform(0, 2 * np.pi, n)
    pts = np.column_stack([np.cos(theta), np.sin(theta)])
    return pts + noise * rng.standard_normal((n, 2))


def make_sphere(n=100, noise=0.02, seed=0):
    """Sample the unit sphere S^2 by normalising Gaussian vectors.

    Each point is obtained by drawing z ~ N(0, I_3) and projecting it
    onto the sphere via x = z / ||z||. This yields the uniform
    distribution on S^2. A small Gaussian perturbation is then added.

    Parameters
    ----------
    n : int
        Number of points.
    noise : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    points : ndarray of shape (n, 3)

    Raises
    ------
    TypeError
        If "n" is not an integer, or "noise" is not a number.
    ValueError
        If "n" is below 2, or "noise" is negative.
    """
    n = _validate_n(n)
    noise = _validate_noise(noise)

    rng = np.random.default_rng(seed)
    z = rng.standard_normal((n, 3))
    pts = z / np.linalg.norm(z, axis=1, keepdims=True)
    return pts + noise * rng.standard_normal((n, 3))


def make_torus(n_u=20, n_v=5, R=3.0, r=1.0, noise=0.02, seed=0):
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
    noise : float
        Standard deviation of the additive Gaussian noise.
    seed : int
        Random seed for the perturbation.

    Returns
    -------
    points : ndarray of shape (n_u * n_v, 3)

    Raises
    ------
    TypeError
        If any argument has the wrong type.
    ValueError
        If "n_u" or "n_v" is below 1, if a radius is not positive, if
        "noise" is negative, or if "r" is not smaller than "R" (which
        would make the torus self-intersect and destroy its topology).
    """
    n_u = _validate_n(n_u, name="n_u", minimum=1)
    n_v = _validate_n(n_v, name="n_v", minimum=1)
    R = _validate_radius(R, "R")
    r = _validate_radius(r, "r")
    noise = _validate_noise(noise)

    if n_u * n_v < 2:
        raise ValueError(
            f"n_u * n_v must be at least 2; got {n_u} * {n_v} = {n_u * n_v}."
        )
    if r >= R:
        raise ValueError(
            f"the minor radius r must be strictly smaller than the major "
            f"radius R; got r={r} and R={R}. Otherwise the surface "
            "self-intersects and is no longer a torus."
        )

    u_vals = np.linspace(0, 2 * np.pi, n_u, endpoint=False)
    v_vals = np.linspace(0, 2 * np.pi, n_v, endpoint=False)
    U, V = np.meshgrid(u_vals, v_vals)
    pts = np.column_stack([
        ((R + r * np.cos(V)) * np.cos(U)).flatten(),
        ((R + r * np.cos(V)) * np.sin(U)).flatten(),
        (r * np.sin(V)).flatten(),
    ])
    rng = np.random.default_rng(seed)
    return pts + noise * rng.standard_normal(pts.shape)


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

    Raises
    ------
    TypeError
        If "n" is not an integer.
    ValueError
        If "n" is below 2.
    """
    n = _validate_n(n)

    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n, 3))
