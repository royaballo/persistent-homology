"""Peak memory benchmark across the four synthetic datasets.

Measures the peak memory of the three reduction strategies as the number
of points grows, on each of the four synthetic datasets, using
:mod:`tracemalloc`. Only our own pipeline is measured: Ripser and GUDHI
allocate in C++, outside Python's allocator, and cannot be measured by the
same tool.

After the runs, an empirical scaling exponent is fitted for each
(dataset, method) pair (least-squares fit of log(peak_mb) versus log(n)),
to check whether the near-n^4 growth expected from the combinatorial size
of the VR complex holds across geometries.

Run with::

    python benchmarks/memory_usage.py

Outputs:
    results/memory_usage.csv:                one row per (dataset, method, n)
    results/scaling_exponents_memory.csv:    one row per (dataset, method)

To target a specific dataset, comment out the others in the DATASETS dict
below.
"""

import numpy as np
import pandas as pd

from _common import measure_memory_ours, results_path
from datasets.synthetic import make_circle, make_cube, make_sphere, make_torus

# Configuration
MAX_DIM = 2
MAX_EPS = 1.5
SIZES = [30, 50, 75, 100]

METHOD_ORDER = ["our_standard", "our_clearing", "our_cohomology"]


# Each generator takes a size n and returns a point cloud of n points.
DATASETS = {
    "Circle": lambda n: make_circle(n=n),
    "Sphere": lambda n: make_sphere(n=n),
    "Torus":  lambda n: make_torus(n_u=max(1, round(n / 5)), n_v=5),
    "Cube":   lambda n: make_cube(n=n),
}


def fit_scaling_exponents(df):
    """Fit one empirical scaling exponent per (dataset, method).

    For each (dataset, method) with at least two valid measurements,
    computes the slope of a least-squares fit of log(peak_mb) against
    log(n), i.e. the exponent p in the empirical law peak ~ n**p.
    Rows where peak_mb is missing (out-of-memory failures) are skipped.

    Returns
    -------
    exponents_df : pandas DataFrame
        Columns: dataset, method, exponent.
    """
    rows = []
    for dataset in df["dataset"].unique():
        sub_ds = df[df["dataset"] == dataset]
        for method in METHOD_ORDER:
            sub = sub_ds[(sub_ds["method"] == method)
                         & sub_ds["peak_mb"].notna()].sort_values("n")
            if len(sub) < 2:
                continue
            log_n = np.log(sub["n"].to_numpy())
            log_m = np.log(sub["peak_mb"].to_numpy())
            slope, _ = np.polyfit(log_n, log_m, 1)
            rows.append({"dataset": dataset, "method": method,
                         "exponent": slope})
    return pd.DataFrame(rows)


def benchmark_dataset(dataset_name, generator):
    """Run the size sweep for a single dataset."""
    results = []

    for n in SIZES:
        print(f"\n[{dataset_name}, n = {n}]")
        pts = generator(n)

        for alg in ["standard", "clearing", "cohomology"]:
            try:
                peak_mb = measure_memory_ours(pts, MAX_DIM, MAX_EPS, alg)
                print(f"   our {alg:<14}: {peak_mb:10.2f} MB")
                results.append({"dataset": dataset_name,
                                "method": f"our_{alg}", "n": n,
                                "peak_mb": peak_mb})
            except MemoryError:
                print(f"   our {alg:<14}: FAILED (out of memory)")
                results.append({"dataset": dataset_name,
                                "method": f"our_{alg}", "n": n,
                                "peak_mb": None})

    return results


def main():
    all_results = []
    for name, generator in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"DATASET: {name}")
        print(f"{'=' * 60}")
        all_results.extend(benchmark_dataset(name, generator))

    df = pd.DataFrame(all_results)
    out = results_path("memory_usage.csv")
    df.to_csv(out, index=False)
    print(f"\nMeasurements: {len(df)} rows written to {out}")

    # Fit and save the scaling exponents, per (dataset, method).
    exponents_df = fit_scaling_exponents(df)
    exp_out = results_path("scaling_exponents_memory.csv")
    exponents_df.to_csv(exp_out, index=False)
    print(f"Scaling exponents written to {exp_out}")
    print("\nEmpirical memory scaling exponents (peak ~ n^p):")
    for _, row in exponents_df.iterrows():
        print(f"  {row['dataset']:<8} {row['method']:<16}: "
              f"p = {row['exponent']:.2f}")


if __name__ == "__main__":
    main()
