"""Inspect protein C-alpha coordinates and compute persistent homology.

Loads the .npy coordinate files produced by "download_proteins.py",
prints a human-readable summary of each, then runs Ripser at the
threshold and persistence cutoff used in the Chapter 6 case study.
Counts of persistent features and the most persistent points are saved
to a CSV for downstream use.
"""

from pathlib import Path

import numpy as np
import pandas as pd  # type: ignore[import-untyped]
from ripser import ripser  # type: ignore[import-not-found]

DATA_DIR = Path("data/proteins")
COORDS_DIR = DATA_DIR / "ca_coords"
RESULTS_DIR = DATA_DIR / "results"

# Filtration threshold used in the chapter, in angstroms.
FILTRATION_THRESHOLD = 12.0
# Below this persistence we consider features as backbone noise.nalyse_protein(pdb_id)
PERSISTENCE_THRESHOLD = 2.0

PROTEINS = ["1AKI", "1EMA", "1A3N"]


def summarise_coordinates(pdb_id, coords):
    """Print a human-readable summary of a C-alpha point cloud."""
    n, _ = coords.shape
    centre = coords.mean(axis=0)
    spread = coords.std(axis=0)
    extent = coords.max(axis=0) - coords.min(axis=0)
    # Diameter is useful to compare to the filtration threshold.
    # Computed via the bounding-box diagonal (an upper bound on the
    # true diameter, fast and good enough for a sanity check).
    diameter_bbox = np.linalg.norm(extent)

    print(f"\n[{pdb_id}] C-alpha coordinates")
    print(f"  shape           : {coords.shape}  (n_residues x dimension)")
    print(f"  centre of mass  : ({centre[0]:.2f}, {centre[1]:.2f}, {centre[2]:.2f}) A")
    print(f"  std per axis    : ({spread[0]:.2f}, {spread[1]:.2f}, {spread[2]:.2f}) A")
    print(f"  bounding box    : {extent[0]:.1f} x {extent[1]:.1f} x {extent[2]:.1f} A")
    print(f"  bbox diameter   : {diameter_bbox:.1f} A "
          f"({'>' if diameter_bbox > FILTRATION_THRESHOLD else '<'} "
          f"{FILTRATION_THRESHOLD} A filtration threshold)")
    print(f"  first 3 atoms   :")
    for i in range(min(3, n)):
        x, y, z = coords[i]
        print(f"    residue {i+1:>3}: ({x:7.2f}, {y:7.2f}, {z:7.2f})")


def count_persistent_features(diagram, persistence_cutoff):
    """Split a persistence diagram into persistent, essential, and noise.

    A "persistent" feature has (death - birth) >= persistence_cutoff and
    a finite death. An "essential" feature has death == infinity (in
    Ripser, this is encoded as np.inf, which means the feature survives
    past the filtration threshold).
    """
    if len(diagram) == 0:
        return {"total": 0, "noise": 0, "persistent": 0, "essential": 0}

    births = diagram[:, 0]
    deaths = diagram[:, 1]
    is_essential = np.isinf(deaths)
    finite = ~is_essential
    persistence = np.where(finite, deaths - births, 0.0)

    n_persistent = int(((persistence >= persistence_cutoff) & finite).sum())
    n_essential = int(is_essential.sum())
    n_noise = int(((persistence < persistence_cutoff) & finite).sum())
    return {
        "total": len(diagram),
        "noise": n_noise,
        "persistent": n_persistent,
        "essential": n_essential,
    }


def top_persistent(diagram, k=5):
    """Return the top-k features by persistence (finite ones only)."""
    if len(diagram) == 0:
        return []
    births = diagram[:, 0]
    deaths = diagram[:, 1]
    finite = ~np.isinf(deaths)
    if not finite.any():
        return []
    persistence = deaths - births
    persistence[~finite] = -np.inf  # exclude essentials from "top finite"
    order = np.argsort(persistence)[::-1][:k]
    return [(float(births[i]), float(deaths[i]),
             float(persistence[i])) for i in order]


def analyse_protein(pdb_id):
    """Load, summarise, and compute persistence for one protein."""
    coords_path = COORDS_DIR / f"{pdb_id}_ca.npy"
    coords = np.load(coords_path)

    summarise_coordinates(pdb_id, coords)

    print(f"\n  Running Ripser (maxdim=2, thresh={FILTRATION_THRESHOLD} A)...")
    result = ripser(coords, maxdim=2, thresh=FILTRATION_THRESHOLD)
    diagrams = result["dgms"]

    rows = []
    for dim, dgm in enumerate(diagrams):
        counts = count_persistent_features(
            dgm, PERSISTENCE_THRESHOLD
        )
        print(f"    H_{dim}: total={counts['total']:>5}  "
              f"noise={counts['noise']:>5}  "
              f"persistent={counts['persistent']:>3}  "
              f"essential={counts['essential']:>3}")
        rows.append({
            "protein": pdb_id,
            "dimension": dim,
            **counts,
        })

        # Show the most persistent features at dim 1 and 2, the
        # topologically interesting ones in the chapter.
        if dim in (1, 2) and len(dgm) > 0:
            top = top_persistent(dgm, k=3)
            for birth, death, pers in top:
                if pers >= PERSISTENCE_THRESHOLD:
                    print(f"        birth={birth:.2f} A, death={death:.2f} A, "
                          f"persistence={pers:.2f} A")

    return rows, diagrams


def main():
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    all_rows = []
    for pdb_id in PROTEINS:
        rows, diagrams = analyse_protein(pdb_id)
        all_rows.extend(rows)

        # Save the raw diagrams for downstream plotting.
        for dim, dgm in enumerate(diagrams):
            out = RESULTS_DIR / f"{pdb_id}_dgm_H{dim}.npy"
            np.save(out, dgm)
        print(f"  Diagrams saved to {RESULTS_DIR}/{pdb_id}_dgm_H*.npy")

    summary = pd.DataFrame(all_rows)
    csv_path = RESULTS_DIR / "betti_counts.csv"
    summary.to_csv(csv_path, index=False)
    print(f"\nSummary table saved to {csv_path}")
    print("\n" + summary.to_string(index=False))


if __name__ == "__main__":
    main()
