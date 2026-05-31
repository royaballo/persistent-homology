"""Helpers for the benchmark scripts.

This module centralises everything the individual benchmark scripts have
in common: making the package and the dataset generators importable,
timing the three reduction strategies and the two reference libraries,
measuring peak memory, and running a measurement under a hard wall-clock
timeout.

Importing this module has the side effect of adding "src/" and
"datasets/" to "sys.path", so benchmark scripts can simply do::

    from _common import time_ours, time_ripser, safe_run_timed
"""

import gc
import signal
import time
import tracemalloc
from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]

from datasets.synthetic import make_circle, make_cube, make_sphere, make_torus
from persistent_homology import compute_persistence

# Reference libraries.
import gudhi  # type: ignore[import-not-found]
from ripser import ripser  # type: ignore[import-not-found]


#### Configuration settings
TIMEOUT_SECONDS = 3600 * 3
N_REPEATS_LIBS = 10
N_REPEATS_OURS = 10
MAX_DIM = 2  # compute H_0, H_1, H_2
MAX_EPS = 1.5
METHOD_ORDER = ["our_standard", "our_clearing", "our_cohomology", "GUDHI", "Ripser"]
DATASETS = {
    "Circle": lambda n: make_circle(n=n),
    "Sphere": lambda n: make_sphere(n=n),
    "Torus": lambda n: make_torus(n_u=max(1, round(n / 5)), n_v=5),
    "Cube": lambda n: make_cube(n=n),
}


# Timing helpers
def _summarise_times(times):
    """Return (median, q1, q3, iqr) over a list of timings."""
    arr = np.asarray(times, dtype=float)
    median = float(np.median(arr))
    q1 = float(np.quantile(arr, 0.25))
    q3 = float(np.quantile(arr, 0.75))
    return median, q1, q3, q3 - q1


def time_gudhi(points, max_dim, max_eps):
    """Time GUDHI's VR persistence.

    Returns
    -------
    median_time : float
    q1, q3, iqr : float
    n_pairs : int
    """
    if gudhi is None:
        raise RuntimeError("GUDHI is not installed")
    times = []
    n_pairs = 0
    for _ in range(N_REPEATS_LIBS):
        t0 = time.perf_counter()
        rips = gudhi.RipsComplex(points=points, max_edge_length=max_eps)
        st = rips.create_simplex_tree(max_dimension=max_dim + 1)
        diag = st.persistence(homology_coeff_field=2)
        times.append(time.perf_counter() - t0)
        n_pairs = len(diag)
    median, q1, q3, iqr = _summarise_times(times)
    return median, q1, q3, iqr, n_pairs


def time_ripser(points, max_dim, max_eps):
    """Time Ripser's VR persistence.

    Returns
    -------
    median_time : float
    q1, q3, iqr : float
    n_pairs : int
    """
    if ripser is None:
        raise RuntimeError("Ripser is not installed")
    times = []
    n_pairs = 0
    for _ in range(N_REPEATS_LIBS):
        t0 = time.perf_counter()
        result = ripser(points, maxdim=max_dim, thresh=max_eps)
        times.append(time.perf_counter() - t0)
        n_pairs = sum(len(d) for d in result["dgms"])
    median, q1, q3, iqr = _summarise_times(times)
    return median, q1, q3, iqr, n_pairs


def time_ours(points, max_dim, max_eps, algorithm):
    """Time one of our three Python reductions.

    For the pure-Python pipeline, runtimes range from milliseconds to
    minutes, so we keep N_REPEATS = 10 for consistency; OS jitter is
    negligible compared to the actual computation.

    Returns
    -------
    median_time : float
    q1, q3, iqr : float
    n_pairs : int
    """
    times = []
    n_pairs = 0
    for _ in range(N_REPEATS_OURS):
        t0 = time.perf_counter()
        diagram = compute_persistence(
            points, max_dim=max_dim + 1, max_eps=max_eps, algorithm=algorithm
        )
        times.append(time.perf_counter() - t0)
        n_pairs = sum(len(v) for v in diagram.values())
    median, q1, q3, iqr = _summarise_times(times)
    return median, q1, q3, iqr, n_pairs


# Memory measurement


def measure_memory_ours(points, max_dim, max_eps, algorithm):
    """Measure the peak memory of our pipeline, in MB.

    Uses :mod:'tracemalloc', which captures Python-level allocations only;
    this matches what our pure-Python pipeline consumes. Garbage
    collection is forced beforehand to ensure a clean baseline.

    Returns
    -------
    peak_mb : float
        Peak traced memory during the computation, in megabytes.
    """
    gc.collect()
    tracemalloc.start()
    compute_persistence(
        points, max_dim=max_dim + 1, max_eps=max_eps, algorithm=algorithm
    )
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_bytes / (1024 * 1024)


def fit_scaling_exponents(df, x_col="n", y_col="time_s"):
    """Fit one empirical scaling exponent per (dataset, method),
    with its standard error.

    For each (dataset, method) with at least two measurements, computes
    the slope of a least-squares fit of log(y) against log(x), i.e. the
    exponent p in the empirical law y ~ x**p. The standard error on the
    slope is derived from the covariance matrix returned by polyfit.

    Parameters
    ----------
    df : pandas DataFrame
        Must contain 'dataset', 'method', x_col, and y_col.
    x_col : str
        Column holding the independent variable (default: 'n').
    y_col : str
        Column holding the dependent variable (default: 'time_s').

    Returns
    -------
    exponents_df : pandas DataFrame
        Columns: dataset, method, exponent, stderr, n_points.
        n_points reports how many measurements went into the fit, so
        the reader knows the reliability of stderr at a glance.
    """
    rows = []
    for dataset in df["dataset"].unique():
        sub_ds = df[df["dataset"] == dataset]
        for method in METHOD_ORDER:
            sub = sub_ds[sub_ds["method"] == method].sort_values(x_col)
            # Drop rows with missing measurements (e.g. MemoryError).
            sub = sub[sub[y_col].notna()]
            sub = sub[sub[y_col] > 0]
            if len(sub) < 2:
                continue

            log_x = np.log(sub[x_col].to_numpy())
            log_y = np.log(sub[y_col].to_numpy())

            # With cov=True, polyfit returns (coeffs, covariance matrix).
            # The standard error on the slope is sqrt of the [0, 0] entry.
            # cov=True requires at least 3 points; for n=2 we report
            # the fit but no stderr.
            if len(sub) >= 3:
                coeffs, cov = np.polyfit(log_x, log_y, 1, cov=True)
                slope = float(coeffs[0])
                stderr = float(np.sqrt(cov[0, 0]))
            else:
                slope = float(np.polyfit(log_x, log_y, 1)[0])
                stderr = float("nan")

            rows.append(
                {
                    "dataset": dataset,
                    "method": method,
                    "exponent": slope,
                    "stderr": stderr,
                    "n_points": len(sub),
                }
            )
    return pd.DataFrame(rows)


# Timeout wrapper


class TimeoutException(Exception):
    """Raised when a measurement exceeds its wall-clock budget."""


def _timeout_handler(signum, frame):
    raise TimeoutException()


def safe_run_timed(label, fn, *args, timeout_s=TIMEOUT_SECONDS):
    """Run a measurement function with a hard timeout and graceful failure.

    Parameters
    ----------
    label : str
        Short name printed alongside the result.
    fn : callable
        A timing function such as :func:'time_ours'; it must return a
        "(time, count)" pair.
    *args
        Positional arguments forwarded to "fn".
    timeout_s : int
        Wall-clock budget in seconds.

    Returns
    -------
    result : tuple or (None, None)
        Whatever "fn" returns, or "(None, None)" on timeout, memory
        exhaustion, or any other error.

    Notes
    -----
    The timeout relies on "SIGALRM" and is therefore only effective on
    UNIX-like systems. On platforms without "SIGALRM" (e.g. Windows)
    the function simply runs to completion.
    """
    use_alarm = hasattr(signal, "SIGALRM")
    if use_alarm:
        signal.signal(signal.SIGALRM, _timeout_handler)
        signal.alarm(timeout_s)
    try:
        result = fn(*args)
        if use_alarm:
            signal.alarm(0)
        t = result[0]
        k = result[-1]
        print(f"   {label:<18}: {t:10.4f}s   ({k} intervals)")
        return result
    except TimeoutException:
        print(f"   {label:<18}: SKIPPED  (exceeds {timeout_s}s budget)")
        return None
    except MemoryError:
        if use_alarm:
            signal.alarm(0)
        print(f"   {label:<18}: FAILED  (out of memory)")
        return None
    except Exception as e:
        if use_alarm:
            signal.alarm(0)
        print(f"   {label:<18}: FAILED  ({type(e).__name__}: {str(e)[:60]})")
        return None


# Output location

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def results_path(filename):
    """Return the path of an output CSV, creating "results/" if needed."""
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / filename
