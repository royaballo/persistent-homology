"""Effect of the filtration scope on computation time, across geometries.

Holds the number of points (n = 100) fixed, and varies the maximum
filtration value max_eps on each of the four synthetic datasets. This
isolates the joint effect of the filtration's reach and of the dataset's
geometry on the cost of the persistence computation, and lets us test
whether the cohomology speedup depends on the local density of the point
cloud.

Run with::

    python benchmarks/effect_filtration_scope.py

Output: results/effect_filtration_scope.csv
        (one row per dataset, method, max_eps)

To target a specific dataset, comment out the others in the DATASETS dict
below.
"""

import pandas as pd

from _common import results_path, safe_run_timed, time_gudhi, time_ours, time_ripser
from datasets.synthetic import make_circle, make_cube, make_sphere, make_torus

# Configuration
MAX_DIM = 2
N = 50
MAX_EPS_VALUES = [0.5, 1.0, 1.5, 2.0, 2.5]


# Each generator takes a size n and returns a point cloud of n points.
DATASETS = {
    "Circle": lambda n: make_circle(n=n),
    "Sphere": lambda n: make_sphere(n=n),
    "Torus":  lambda n: make_torus(n_u=max(1, round(n / 5)), n_v=5),
    "Cube":   lambda n: make_cube(n=n),
}


def benchmark_dataset(dataset_name, generator):
    """Sweep over max_eps for a single dataset, at fixed n."""
    results = []

    # The point cloud is fixed across the eps sweep.
    pts = generator(N)

    for eps in MAX_EPS_VALUES:
        print(f"\n[{dataset_name}, max_eps = {eps}]")

        for alg in ["standard", "clearing", "cohomology"]:
            t, k = safe_run_timed(f"our {alg}", time_ours,
                                  pts, MAX_DIM, eps, alg)
            if t is not None:
                results.append({"dataset": dataset_name,
                                "method": f"our_{alg}",
                                "max_eps": eps,
                                "time_s": t, "pairs": k})

        t, k = safe_run_timed("GUDHI", time_gudhi, pts, MAX_DIM, eps)
        if t is not None:
            results.append({"dataset": dataset_name, "method": "GUDHI",
                            "max_eps": eps, "time_s": t, "pairs": k})

        t, k = safe_run_timed("Ripser", time_ripser, pts, MAX_DIM, eps)
        if t is not None:
            results.append({"dataset": dataset_name, "method": "Ripser",
                            "max_eps": eps, "time_s": t, "pairs": k})

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
