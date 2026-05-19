"""Peak memory benchmark on the noisy circle.

Measures the peak memory of the three reduction strategies as the number
of points grows, using :mod:`tracemalloc`. Only our own pipeline is
measured: Ripser and GUDHI allocate in C++, outside Python's allocator,
and cannot be measured by the same tool.

Run with::

    python benchmarks/memory_usage.py

Output: results/memory_usage.csv
"""

import pandas as pd

from _common import measure_memory_ours, results_path
from datasets.synthetic import make_circle

# Configuration
MAX_DIM = 2
MAX_EPS = 2.0
SIZES = [30, 50, 75, 100]


def main():
    results = []

    for n in SIZES:
        print(f"\n[n = {n}]")
        pts = make_circle(n=n)

        for alg in ["standard", "clearing", "cohomology"]:
            try:
                peak_mb = measure_memory_ours(pts, MAX_DIM, MAX_EPS, alg)
                print(f"   our {alg:<14}: {peak_mb:10.2f} MB")
                results.append({"method": f"our_{alg}", "n": n,
                                "peak_mb": peak_mb})
            except MemoryError:
                print(f"   our {alg:<14}: FAILED (out of memory)")
                results.append({"method": f"our_{alg}", "n": n,
                                "peak_mb": None})

    df = pd.DataFrame(results)
    out = results_path("memory_usage.csv")
    df.to_csv(out, index=False)
    print(f"\nDone. {len(df)} measurements written to {out}")


if __name__ == "__main__":
    main()
