"""Computation time benchmark on the noisy circle.

Measures the wall-clock time of the three reduction strategies and of the
two reference libraries as the number of points grows. Our pure-Python
pipeline is tested up to n = 100; the libraries extend further. GUDHI is
capped at n = 200, beyond which it exhausts memory on the reference
machine.

After the timing runs, an empirical scaling exponent is fitted for each
method (least-squares fit of log(time) versus log(n)). The exponents are
a result in their own right, they appear in the experimental chapter,
so they are computed here and saved alongside the timings, rather than
recomputed at plotting time.

Run with::

    python benchmarks/computation_time.py

Outputs:
    results/computation_time.csv: one row per (method, n) measurement
    results/scaling_exponents.csv: one row per method
"""

import numpy as np
import pandas as pd

from _common import results_path, safe_run_timed, time_gudhi, time_ours, time_ripser
from datasets.synthetic import make_circle

# Configuration
MAX_DIM = 2                       # compute H_0, H_1, H_2
MAX_EPS = 2.0
SIZES_OURS = [30, 50, 75, 100]            # our pure-Python pipeline
SIZES_GUDHI = [30, 50, 75, 100, 200]      # GUDHI: capped at 200 (OOM at 500)
SIZES_RIPSER = [30, 50, 75, 100, 200, 500]  # Ripser: full range

METHOD_ORDER = ["our_standard", "our_clearing", "our_cohomology",
                "GUDHI", "Ripser"]


def fit_scaling_exponents(df):
    """Fit an empirical scaling exponent for each method.

    For a method timed at sizes "n" with times "t", the exponent is
    the slope of a least-squares fit of "log(t)" against "log(n)",
    i.e. the exponent "p" in the empirical law "t ~ n**p".

    Parameters
    ----------
    df : pandas DataFrame
        The timing results, with columns "method", "n", "time_s".

    Returns
    -------
    exponents_df : pandas DataFrame
        One row per method, with columns "method" and "exponent".
        Methods with fewer than two data points are skipped.
    """
    rows = []
    for method in METHOD_ORDER:
        sub = df[df["method"] == method].sort_values("n")
        if len(sub) < 2:
            continue
        log_n = np.log(sub["n"].to_numpy())
        log_t = np.log(sub["time_s"].to_numpy())
        slope, _ = np.polyfit(log_n, log_t, 1)
        rows.append({"method": method, "exponent": slope})
    return pd.DataFrame(rows)


def main():
    results = []

    # Pre-generate the point clouds once, shared across methods.
    clouds = {n: make_circle(n=n) for n in SIZES_RIPSER}

    for n in SIZES_RIPSER:
        print(f"\n[n = {n}]")
        pts = clouds[n]

        if n in SIZES_OURS:
            for alg in ["standard", "clearing", "cohomology"]:
                t, k = safe_run_timed(f"our {alg}", time_ours,
                                      pts, MAX_DIM, MAX_EPS, alg)
                if t is not None:
                    results.append({"method": f"our_{alg}", "n": n,
                                    "time_s": t, "pairs": k})

        if n in SIZES_GUDHI:
            t, k = safe_run_timed("GUDHI", time_gudhi,
                                  pts, MAX_DIM, MAX_EPS)
            if t is not None:
                results.append({"method": "GUDHI", "n": n,
                                "time_s": t, "pairs": k})

        t, k = safe_run_timed("Ripser", time_ripser,
                              pts, MAX_DIM, MAX_EPS)
        if t is not None:
            results.append({"method": "Ripser", "n": n,
                            "time_s": t, "pairs": k})

    df = pd.DataFrame(results)
    out = results_path("computation_time.csv")
    df.to_csv(out, index=False)
    print(f"\nTimings: {len(df)} measurements written to {out}")

    # Fit and save the scaling exponents.
    exponents_df = fit_scaling_exponents(df)
    exp_out = results_path("scaling_exponents.csv")
    exponents_df.to_csv(exp_out, index=False)
    print(f"Scaling exponents written to {exp_out}")
    print("\nEmpirical scaling exponents (t ~ n^p):")
    for _, row in exponents_df.iterrows():
        print(f"  {row['method']:<16}: p = {row['exponent']:.2f}")


if __name__ == "__main__":
    main()
