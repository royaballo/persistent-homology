"""Effect of dataset geometry on computation time.

Holds the number of points (n = 100) and the filtration scope
(max_eps = 2.0) fixed, and varies the dataset across the four synthetic
geometries. This isolates the effect of how the points are distributed
in space on the cost of the persistence computation.

Run with::

    python benchmarks/effect_geometry.py

Output: results/effect_geometry.csv
"""

import pandas as pd

from _common import results_path, safe_run_timed, time_gudhi, time_ours, time_ripser
from datasets.synthetic import make_circle, make_cube, make_sphere, make_torus

# Configuration
MAX_DIM = 2
MAX_EPS = 2.0
N = 100

# Datasets, each fixed at n = 100 points.
DATASETS = {
    "Circle": make_circle(n=N),
    "Sphere": make_sphere(n=N),
    "Torus": make_torus(n_u=20, n_v=5),   # 20 * 5 = 100 points
    "Cube": make_cube(n=N),
}


def main():
    results = []

    for name, pts in DATASETS.items():
        print(f"\n[{name}]  n = {len(pts)}, max_eps = {MAX_EPS}")

        for alg in ["standard", "clearing", "cohomology"]:
            t, k = safe_run_timed(f"our {alg}", time_ours,
                                  pts, MAX_DIM, MAX_EPS, alg)
            if t is not None:
                results.append({"method": f"our_{alg}", "dataset": name,
                                "time_s": t, "pairs": k})

        t, k = safe_run_timed("GUDHI", time_gudhi, pts, MAX_DIM, MAX_EPS)
        if t is not None:
            results.append({"method": "GUDHI", "dataset": name,
                            "time_s": t, "pairs": k})

        t, k = safe_run_timed("Ripser", time_ripser, pts, MAX_DIM, MAX_EPS)
        if t is not None:
            results.append({"method": "Ripser", "dataset": name,
                            "time_s": t, "pairs": k})

    df = pd.DataFrame(results)
    out = results_path("effect_geometry.csv")
    df.to_csv(out, index=False)
    print(f"\nDone. {len(df)} measurements written to {out}")


if __name__ == "__main__":
    main()
