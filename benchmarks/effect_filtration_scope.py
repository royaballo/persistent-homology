"""Effect of the filtration scope on computation time, across geometries.

Holds the number of points (n = 100) fixed, and varies the maximum
filtration value max_eps on each of the four synthetic datasets. This
isolates the joint effect of the filtration's reach and of the dataset's
geometry on the cost of the persistence computation, and lets us test
whether the cohomology speedup depends on the local density of the point
cloud.

Each measurement records the median, Q1, Q3 and IQR over N_REPEATS runs,
following the methodology refinement of Section 5.1.2.

Run with::

    python benchmarks/effect_filtration_scope.py

Output: results/effect_filtration_scope.csv
        (one row per dataset, method, max_eps)

To target a specific dataset, comment out the others in the DATASETS dict
below.
"""

import pandas as pd  # type: ignore[import-untyped]
from _common import (
    DATASETS,
    results_path,
    safe_run_timed,
    time_gudhi,
    time_ours,
    time_ripser,
)

# Configuration
MAX_DIM = 2  # compute H_0, H_1, H_2
N = 50
MAX_EPS_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]

def benchmark_dataset(dataset_name, generator):
    """Sweep over max_eps for a single dataset, at fixed n.

    Each measurement records median, Q1, Q3 and IQR over N_REPEATS
    runs of the timing routine.
    """
    results = []

    # The point cloud is fixed across the eps sweep.
    pts = generator(N)

    for eps in MAX_EPS_VALUES:
        print(f"\n[{dataset_name}, max_eps = {eps}]")

        for alg in ["standard", "clearing", "cohomology"]:
            result = safe_run_timed(f"our {alg}", time_ours, pts, MAX_DIM, eps, alg)
            if result is not None:
                median, q1, q3, iqr, k = result
                results.append(
                    {
                        "dataset": dataset_name,
                        "method": f"our_{alg}",
                        "max_eps": eps,
                        "time_s": median,
                        "q1_s": q1,
                        "q3_s": q3,
                        "iqr_s": iqr,
                        "pairs": k,
                    }
                )

        result = safe_run_timed("GUDHI", time_gudhi, pts, MAX_DIM, eps)
        if result is not None:
            median, q1, q3, iqr, k = result
            results.append(
                {
                    "dataset": dataset_name,
                    "method": "GUDHI",
                    "max_eps": eps,
                    "time_s": median,
                    "q1_s": q1,
                    "q3_s": q3,
                    "iqr_s": iqr,
                    "pairs": k,
                }
            )

        result = safe_run_timed("Ripser", time_ripser, pts, MAX_DIM, eps)
        if result is not None:
            median, q1, q3, iqr, k = result
            results.append(
                {
                    "dataset": dataset_name,
                    "method": "Ripser",
                    "max_eps": eps,
                    "time_s": median,
                    "q1_s": q1,
                    "q3_s": q3,
                    "iqr_s": iqr,
                    "pairs": k,
                }
            )

    return results


def main():
    all_results = []
    for name, generator in DATASETS.items():
        print(f"\n{'=' * 60}")
        print(f"DATASET: {name}")
        print(f"{'=' * 60}")
        all_results.extend(benchmark_dataset(name, generator))

    df = pd.DataFrame(all_results)
    out = results_path("effect_filtration_scope.csv")
    df.to_csv(out, index=False)
    print(f"\nDone. {len(df)} measurements written to {out}")


if __name__ == "__main__":
    main()
