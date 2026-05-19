"""Effect of the filtration scope on computation time.

Holds the dataset (the noisy circle) and the number of points (n = 100)
fixed, and varies the maximum filtration value max_eps. This isolates the
effect of the filtration's reach -- and hence of the density of the
Vietoris-Rips complex -- on the cost of the persistence computation.

Run with::

    python benchmarks/effect_filtration_scope.py

Output: results/effect_filtration_scope.csv
"""

import pandas as pd

from _common import results_path, safe_run_timed, time_gudhi, time_ours, time_ripser
from datasets.synthetic import make_circle

# Configuration
MAX_DIM = 2
N = 100
MAX_EPS_VALUES = [1.0, 1.5, 2.0, 2.5]


def main():
    results = []

    # The point cloud is fixed; only the filtration scope varies.
    pts = make_circle(n=N)

    for eps in MAX_EPS_VALUES:
        print(f"\n[max_eps = {eps}]")

        for alg in ["standard", "clearing", "cohomology"]:
            t, k = safe_run_timed(f"our {alg}", time_ours,
                                  pts, MAX_DIM, eps, alg)
            if t is not None:
                results.append({"method": f"our_{alg}", "max_eps": eps,
                                "time_s": t, "pairs": k})

        t, k = safe_run_timed("GUDHI", time_gudhi, pts, MAX_DIM, eps)
        if t is not None:
            results.append({"method": "GUDHI", "max_eps": eps,
                            "time_s": t, "pairs": k})

        t, k = safe_run_timed("Ripser", time_ripser, pts, MAX_DIM, eps)
        if t is not None:
            results.append({"method": "Ripser", "max_eps": eps,
                            "time_s": t, "pairs": k})

    df = pd.DataFrame(results)
    out = results_path("effect_filtration_scope.csv")
    df.to_csv(out, index=False)
    print(f"\nDone. {len(df)} measurements written to {out}")


if __name__ == "__main__":
    main()
