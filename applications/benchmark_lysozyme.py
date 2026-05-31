"""Benchmark all five implementations on Lysozyme (1AKI).

Loads the C-alpha coordinates extracted by "download_proteins.py",
then runs our standard, clearing, and cohomology reductions alongside
Ripser and GUDHI. Reports the median wall-clock time over N_RUNS,
and verifies that all implementations recover the same Betti numbers.

This script is the Chapter 6 counterpart of the synthetic-data
benchmarks of Chapter 5: it asks whether our Python implementations
deliver the same relative performance ordering on real biological data.
Only Lysozyme (n = 129) is benchmarked with our Python implementations,
because larger proteins exceed the practical reach of the
explicit-matrix pipeline (see Chapter 5).
"""

from pathlib import Path
import time

import gudhi  # type: ignore[import-not-found]
import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from ripser import ripser  # type: ignore[import-not-found]

from persistent_homology import compute_persistence

PROTEIN = "1AKI"
COORDS_PATH = Path("data/proteins/ca_coords") / f"{PROTEIN}_ca.npy"
RESULTS_DIR = Path("data/proteins/results")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

FILTRATION_THRESHOLD = 12.0   # angstroms, as in Chapter 6
PERSISTENCE_THRESHOLD = 2.0   # angstroms
MAX_DIM = 2                   # H_0, H_1, H_2
N_RUNS = 10


def median_time(func, n_runs=N_RUNS):
    """Run "func" n_runs times, return (median wall-clock time, last result)."""
    times = []
    result = None
    for _ in range(n_runs):
        t0 = time.perf_counter()
        result = func()
        times.append(time.perf_counter() - t0)
    return float(np.median(times)), result


def count_betti(diagrams_by_dim, persistence_threshold):
    """Return [b0, b1, b2] from a list-of-arrays persistence diagram.

    A feature contributes to the Betti count when its persistence
    (death - birth) is at least "persistence_threshold", or when it
    is essential (death == inf).
    """
    betti = []
    for dim in range(MAX_DIM + 1):
        dgm = diagrams_by_dim[dim]
        if len(dgm) == 0:
            betti.append(0)
            continue
        births = dgm[:, 0]
        deaths = dgm[:, 1]
        essential = np.isinf(deaths)
        finite = ~essential
        persistent = (deaths - births >= persistence_threshold) & finite
        betti.append(int(persistent.sum() + essential.sum()))
    return betti


def run_ours(points, algorithm):
    """Run one of our three Python implementations.

    "compute_persistence" returns a dict keyed by dimension; we
    convert it to a list of (n, 2) arrays indexed by dimension so the
    downstream code is uniform across implementations.
    """
    diag_dict = compute_persistence(
        points,
        max_dim=MAX_DIM + 1,         # need one above MAX_DIM internally
        max_eps=FILTRATION_THRESHOLD,
        algorithm=algorithm,
    )
    diagrams = []
    for dim in range(MAX_DIM + 1):
        intervals = diag_dict.get(dim, [])
        if len(intervals) == 0:
            diagrams.append(np.zeros((0, 2)))
        else:
            diagrams.append(np.array(intervals, dtype=float))
    return diagrams


def run_ripser(points):
    """Run Ripser, return a list of (n, 2) arrays indexed by dimension."""
    res = ripser(points, maxdim=MAX_DIM, thresh=FILTRATION_THRESHOLD)
    return res["dgms"]


def run_gudhi(points):
    """Run GUDHI on the same point cloud, returning the same shape.

    GUDHI's RipsComplex / SimplexTree workflow needs explicit
    persistence() calls. We then bucket the output by dimension to
    match the shape of the other implementations.
    """
    rips = gudhi.RipsComplex(points=points,
                             max_edge_length=FILTRATION_THRESHOLD)
    st = rips.create_simplex_tree(max_dimension=MAX_DIM + 1)
    st.compute_persistence()

    buckets = {dim: [] for dim in range(MAX_DIM + 1)}
    for dim, (birth, death) in st.persistence():
        if dim > MAX_DIM:
            continue
        buckets[dim].append((birth, death))

    diagrams = []
    for dim in range(MAX_DIM + 1):
        if buckets[dim]:
            diagrams.append(np.array(buckets[dim], dtype=float))
        else:
            diagrams.append(np.zeros((0, 2)))
    return diagrams


def main():
    if not COORDS_PATH.exists():
        raise FileNotFoundError(
            f"Coordinates not found at {COORDS_PATH}. "
            f"Run download_proteins.py first."
        )

    coords = np.load(COORDS_PATH)
    print(f"Loaded {PROTEIN}: {coords.shape[0]} C-alpha atoms\n")

    implementations = [
        ("Our standard",   lambda: run_ours(coords, "standard")),
        ("Our clearing",   lambda: run_ours(coords, "clearing")),
        ("Our cohomology", lambda: run_ours(coords, "cohomology")),
        ("Ripser",         lambda: run_ripser(coords)),
        ("GUDHI",          lambda: run_gudhi(coords)),
    ]

    rows = []
    for name, fn in implementations:
        print(f"Running {name}...", flush=True)
        t, diagrams = median_time(fn)
        betti = count_betti(diagrams, PERSISTENCE_THRESHOLD)
        print(f"  time = {t:.3f} s (median of {N_RUNS} runs)")
        print(f"  Betti = (b0={betti[0]}, b1={betti[1]}, b2={betti[2]})")
        rows.append({
            "implementation": name,
            "time_s": t,
            "b0": betti[0],
            "b1": betti[1],
            "b2": betti[2],
        })

    df = pd.DataFrame(rows)

    # Check Betti consistency across implementations.
    betti_columns = ["b0", "b1", "b2"]
    if df[betti_columns].drop_duplicates().shape[0] == 1:
        print("\nAll implementations agree on Betti numbers.")
    else:
        print("\nWARNING: implementations disagree on Betti numbers:")
        print(df[["implementation"] + betti_columns].to_string(index=False))

    # Speedups, taking Ripser as the reference (fastest in Chapter 5).
    ripser_time = df.loc[df.implementation == "Ripser", "time_s"].iloc[0]
    df["speedup_vs_ripser"] = df["time_s"] / ripser_time

    out = RESULTS_DIR / "lysozyme_benchmark.csv"
    df.to_csv(out, index=False)
    print(f"\nResults saved to {out}")
    print("\nSummary:")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
