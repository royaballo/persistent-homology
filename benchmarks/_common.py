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


from persistent_homology import compute_persistence

# Reference libraries.
try:
    import gudhi  # type: ignore[import-not-found]
except ImportError:
    gudhi = None

try:
    from ripser import ripser  # type: ignore[import-not-found]
except ImportError:
    ripser = None


# Default hard timeout for a single measurement (3 hours).
TIMEOUT_SECONDS = 3600 * 3

# Number of repetitions; the median is reported.
N_REPEATS = 3



# Timing helpers

def time_ours(points, max_dim, max_eps, algorithm):
    """Measure the wall-clock time of our pipeline.

    Parameters
    ----------
    points : ndarray of shape (n, d)
    max_dim : int
        Homology is computed up to dimension "max_dim"; internally the
        pipeline is called with "max_dim + 1".
    max_eps : float
    algorithm : {'standard', 'clearing', 'cohomology'}

    Returns
    -------
    median_time : float
        Median wall-clock time over "N_REPEATS" runs, in seconds.
    n_intervals : int
        Number of persistence intervals in the resulting diagram.
    """
    times = []
    diagram = None
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        diagram = compute_persistence(points, max_dim=max_dim + 1,
                                      max_eps=max_eps, algorithm=algorithm)
        times.append(time.perf_counter() - t0)
    n_intervals = sum(len(v) for v in diagram.values())
    return float(np.median(times)), n_intervals


def time_gudhi(points, max_dim, max_eps):
    """Measure GUDHI's Vietoris-Rips persistence (median of N_REPEATS runs).

    Returns
    -------
    median_time : float
    n_pairs : int
    """
    if gudhi is None:
        raise RuntimeError("GUDHI is not installed")
    times = []
    n_pairs = 0
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        rips = gudhi.RipsComplex(points=points, max_edge_length=max_eps)
        st = rips.create_simplex_tree(max_dimension=max_dim + 1)
        diag = st.persistence(homology_coeff_field=2)
        times.append(time.perf_counter() - t0)
        n_pairs = len(diag)
    return float(np.median(times)), n_pairs


def time_ripser(points, max_dim, max_eps):
    """Measure Ripser's Vietoris-Rips persistence (median of N_REPEATS runs).

    Returns
    -------
    median_time : float
    n_pairs : int
    """
    if ripser is None:
        raise RuntimeError("Ripser is not installed")
    times = []
    n_pairs = 0
    for _ in range(N_REPEATS):
        t0 = time.perf_counter()
        result = ripser(points, maxdim=max_dim, thresh=max_eps)
        times.append(time.perf_counter() - t0)
        n_pairs = sum(len(d) for d in result["dgms"])
    return float(np.median(times)), n_pairs



# Memory measurement

def measure_memory_ours(points, max_dim, max_eps, algorithm):
    """Measure the peak memory of our pipeline, in MB.

    Uses :mod:`tracemalloc`, which captures Python-level allocations only;
    this matches what our pure-Python pipeline consumes. Garbage
    collection is forced beforehand to ensure a clean baseline.

    Returns
    -------
    peak_mb : float
        Peak traced memory during the computation, in megabytes.
    """
    gc.collect()
    tracemalloc.start()
    compute_persistence(points, max_dim=max_dim + 1,
                        max_eps=max_eps, algorithm=algorithm)
    _, peak_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return peak_bytes / (1024 * 1024)



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
        A timing function such as :func:`time_ours`; it must return a
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
        t, k = result
        print(f"   {label:<18}: {t:10.4f}s   ({k} intervals)")
        return result
    except TimeoutException:
        print(f"   {label:<18}: SKIPPED  (exceeds {timeout_s}s budget)")
        return None, None
    except MemoryError:
        if use_alarm:
            signal.alarm(0)
        print(f"   {label:<18}: FAILED  (out of memory)")
        return None, None
    except Exception as e:  # noqa: BLE001
        if use_alarm:
            signal.alarm(0)
        print(f"   {label:<18}: FAILED  ({type(e).__name__}: {str(e)[:60]})")
        return None, None


# Output location

RESULTS_DIR = Path(__file__).resolve().parent / "results"


def results_path(filename):
    """Return the path of an output CSV, creating "results/" if needed."""
    RESULTS_DIR.mkdir(exist_ok=True)
    return RESULTS_DIR / filename
