"""Download protein structures from the RCSB Protein Data Bank.

Downloads the three proteins used in the Chapter 6 case study:

* 1AKI: Lysozyme (single-domain enzyme, 129 residues)
* 1EMA: Green Fluorescent Protein (beta-barrel, 221 residues)
* 1A3N: Hemoglobin (tetramer, 572 residues)

The PDB files are cached locally; rerunning the script does not
re-download them. The script also extracts the C-alpha coordinates and
saves them as plain .npy arrays for downstream use.
"""

from pathlib import Path

import numpy as np

from Bio.PDB import PDBList, PDBParser  # type: ignore[import-not-found]

# Where to store the downloaded files and the extracted coordinates.
DATA_DIR = Path("data/proteins")
PDB_DIR = DATA_DIR / "pdb"
COORDS_DIR = DATA_DIR / "ca_coords"

# The three proteins used in the case study, with metadata for reference.
PROTEINS = [
    {"id": "1AKI", "name": "Lysozyme", "expected_residues": 129},
    {"id": "1EMA", "name": "Green Fluorescent Protein", "expected_residues": 221},
    {"id": "1A3N", "name": "Hemoglobin", "expected_residues": 572},
]


def download_pdb(pdb_id, target_dir):
    """Download a PDB file via Biopython, return the local path.

    Biopython caches the file by default; if the file is already present
    it will not re-download. The filename pattern produced is
    ``pdb{pdb_id_lower}.ent``; we rename it to ``{pdb_id}.pdb`` for
    clarity.
    """
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{pdb_id}.pdb"

    if target.exists():
        print(f"  {pdb_id}: already present at {target}")
        return target

    pdbl = PDBList(verbose=False)
    # retrieve_pdb_file writes to pdir as 'pdb{id_lower}.ent' (or .cif).
    raw = pdbl.retrieve_pdb_file(
        pdb_id,
        pdir=str(target_dir),
        file_format="pdb",
        overwrite=False,
    )
    # Rename for consistency.
    Path(raw).rename(target)
    print(f"  {pdb_id}: downloaded to {target}")
    return target


def extract_ca_coordinates(pdb_path, pdb_id):
    """Return an (n, 3) array of C-alpha coordinates from a PDB file.

    Only the first model is used (most PDB files have one model;
    NMR ensembles have several, of which we keep the first). All chains
    are included, so multi-chain proteins like hemoglobin yield the
    union of C-alpha atoms across subunits. Non-standard residues and
    HETATM records are skipped.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure(pdb_id, str(pdb_path))

    coords = []
    model = next(structure.get_models())  # first model only
    for chain in model:
        for residue in chain:
            # Skip waters, ligands, etc. Standard residues have
            # hetflag == ' '.
            if residue.id[0] != " ":
                continue
            if "CA" in residue:
                coords.append(residue["CA"].get_coord())

    if not coords:
        raise RuntimeError(f"No C-alpha atoms found in {pdb_path}")
    return np.array(coords, dtype=np.float64)


def main():
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    COORDS_DIR.mkdir(parents=True, exist_ok=True)

    for protein in PROTEINS:
        pdb_id = protein["id"]
        print(f"\n[{pdb_id}] {protein['name']}")

        pdb_path = download_pdb(pdb_id, PDB_DIR)
        coords = extract_ca_coordinates(pdb_path, pdb_id)

        n = coords.shape[0]
        expected = protein["expected_residues"]
        status = "OK" if n == expected else f"WARNING: expected {expected}"
        print(f"  Extracted {n} C-alpha atoms  [{status}]")

        out = COORDS_DIR / f"{pdb_id}_ca.npy"
        np.save(out, coords)
        print(f"  Saved coordinates to {out}")


if __name__ == "__main__":
    main()
