"""Computation time benchmark across the four synthetic datasets.

Measures the wall-clock time of the three reduction strategies and of the
two reference libraries as the number of points grows, on each of the four
synthetic datasets. Our pure-Python pipeline is tested up to n = 100; the
libraries extend further. GUDHI is capped at n = 200, beyond which it
exhausts memory on the reference machine.

After the timing runs, an empirical scaling exponent is fitted for each
(dataset, method) pair (least-squares fit of log(time) versus log(n)).

Run with::

    python benchmarks/computation_time.py

Outputs:
    results/computation_time.csv:   one row per (dataset, method, n)
    results/scaling_exponents.csv:  one row per (dataset, method)

To target a specific dataset, comment out the others in the DATASETS dict
below.
"""

import numpy as np
import pandas as pd

from _common import results_path, safe_run_timed, time_gudhi, time_ours, time_ripser
from datasets.synthetic import make_circle, make_cube, make_sphere, make_torus

# Configuration
MAX_DIM = 2                               # compute H_0, H_1, H_2
MAX_EPS = 1.5
SIZES_OURS = [30, 50, 75, 100]            # our pure-Python pipeline
SIZES_GUDHI = [30, 50, 75, 100, 200]      # GUDHI: capped at 200 (OOM at 500)
SIZES_RIPSER = [30, 50, 75, 100, 200, 500]  # Ripser: full range

METHOD_ORDER = ["our_standard", "our_clearing", "our_cohomology",
                "GUDHI", "Ripser"]


# Each generator takes a size n and returns a point cloud of n points.
DATASETS = {
    "Circle": lambda n: make_circle(n=n),
    "Sphere": lambda n: make_sphere(n=n),
    "Torus":  lambda n: make_torus(n_u=max(1, round(n / 5)), n_v=5),
    "Cube":   lambda n: make_cube(n=n),
}


def fit_scaling_exponents(df):
    """Fit one empirical scaling exponent per (dataset, method).

    For each (dataset, method) with at least two measurements, computes
    the slope of a least-squares fit of log(t) against log(n), i.e. the
    exponent p in the empirical law t ~ n**p.

    Returns
    -------
    exponents_df : pandas DataFrame
        Columns: dataset, method, exponent.
    """
    rows = []
    for dataset in df["dataset"].unique():
        sub_ds = df[df["dataset"] == dataset]
        for method in METHOD_ORDER:
            sub = sub_ds[sub_ds["method"] == method].sort_values("n")
            if len(sub) < 2:
                continue
            log_n = np.log(sub["n"].to_numpy())
            log_t = np.log(sub["time_s"].to_numpy())
            slope, _ = np.polyfit(log_n, log_t, 1)
            rows.append({"dataset": dataset, "method": method,
                         "exponent": slope})
    return pd.DataFrame(rows)


def benchmark_dataset(dataset_name, generator):
    """Run the full size sweep for a single dataset."""
    results = []

    # Pre-generate the point clouds for this dataset, shared across methods.
    clouds = {n: generator(n) for n in SIZES_RIPSER}

    for n in SIZES_RIPSER:
        print(f"\n[{dataset_name}, n = {n}]")
        pts = clouds[n]

        if n in SIZES_OURS:
            for alg in ["standard", "clearing", "cohomology"]:
                t, k = safe_run_timed(f"our {alg}", time_ours,
                                      pts, MAX_DIM, MAX_EPS, alg)
                if t is not None:
                    results.append({"dataset": dataset_name,
                                    "method": f"our_{alg}", "n": n,
                                    "time_s": t, "pairs": k})

        if n in SIZES_GUDHI:
            t, k = safe_run_timed("GUDHI", time_gudhi,
                                  pts, MAX_DIM, MAX_EPS)
            if t is not None:
                results.append({"dataset": dataset_name, "method": "GUDHI",
                                "n": n, "time_s": t, "pairs": k})

        t, k = safe_run_timed("Ripser", time_ripser,
                              pts, MAX_DIM, MAX_EPS)
        if t is not None:
            results.append({"dataset": dataset_name, "method": "Ripser",
                            "n": n, "time_s": t, "pairs": k})

    return results


def main():
    all_results = []
    for name, generator in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"DATASET: {name}")
        print(f"{'=' * 60}")
        all_results.extend(benchmark_dataset(name, generator))

    df = pd.DataFrame(all_results)
    out = results_path("computation_time.csv")
    df.to_csv(out, index=False)
    print(f"\nTimings: {len(df)} measurements written to {out}")

    # Fit and save the scaling exponents, per (dataset, method).
    exponents_df = fit_scaling_exponents(df)
    exp_out = results_path("scaling_exponents.csv")
    exponents_df.to_csv(exp_out, index=False)
    print(f"Scaling exponents written to {exp_out}")
    print("\nEmpirical scaling exponents (t ~ n^p):")
    for _, row in exponents_df.iterrows():
        print(f"  {row['dataset']:<8} {row['method']:<16}: "
              f"p = {row['exponent']:.2f}")


if __name__ == "__main__":
    main()
