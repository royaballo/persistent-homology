"""Protein structures from the RCSB Protein Data Bank.

Downloads and parses the three proteins used in the Chapter 6 case study:

* 1AKI: Lysozyme (single-domain enzyme, 129 residues)
* 1EMA: Green Fluorescent Protein (beta-barrel, 221 residues)
* 1A3N: Haemoglobin (tetramer, 572 residues)

The PDB files are cached locally, so rerunning does not re-download them.
The C-alpha coordinates are extracted and saved as plain ``.npy`` arrays.

Biopython is required here but not by the rest of the package, so it is
imported lazily inside the functions that need it. Install it with::

    pip install "persistent-homology[bio]"
"""

from pathlib import Path

import numpy as np

# Default location for the downloaded files and extracted coordinates,
# relative to the working directory.
DATA_DIR = Path("data/proteins")
PDB_DIR = DATA_DIR / "pdb"
COORDS_DIR = DATA_DIR / "ca_coords"

# The three proteins used in the case study, with metadata for reference.
PROTEINS = [
    {"id": "1AKI", "name": "Lysozyme", "expected_residues": 129},
    {"id": "1EMA", "name": "Green Fluorescent Protein", "expected_residues": 221},
    {"id": "1A3N", "name": "Haemoglobin", "expected_residues": 572},
]

_BIOPYTHON_HINT = (
    "biopython is required to download and parse PDB files, but it is not "
    "installed. It is an optional dependency of this package; install it "
    'with: pip install "persistent-homology[bio]"'
)


def _require_biopython():
    """Import Biopython, or raise an ImportError that says how to get it.

    Biopython lives behind the ``[bio]`` extra, so a plain install of the
    package does not have it. Importing it at module level would make
    ``import persistent_homology.datasets.proteins`` fail outright for
    every user who never touches proteins; importing it here defers the
    failure to the moment it actually matters, and attaches the fix.

    Raises
    ------
    ImportError
        If Biopython is not installed.
    """
    try:
        from Bio.PDB import PDBList, PDBParser  # type: ignore[import-not-found]
    except ImportError as exc:
        raise ImportError(_BIOPYTHON_HINT) from exc
    return PDBList, PDBParser


def download_pdb(pdb_id, target_dir=None):
    """Download a PDB file, returning the local path.

    The file is cached: if it is already present it is not re-downloaded.

    Parameters
    ----------
    pdb_id : str
        A four-character PDB identifier, such as ``"1AKI"``.
    target_dir : path-like, optional
        Where to store the file. Defaults to :data:`PDB_DIR`.

    Returns
    -------
    path : Path
        The local path to the downloaded ``{pdb_id}.pdb``.

    Raises
    ------
    ImportError
        If Biopython is not installed.
    ValueError
        If ``pdb_id`` is not a four-character identifier.
    RuntimeError
        If the download fails or the file cannot be retrieved.
    """
    PDBList, _ = _require_biopython()

    if not isinstance(pdb_id, str) or len(pdb_id) != 4:
        raise ValueError(
            f"pdb_id must be a four-character PDB identifier such as '1AKI'; "
            f"got {pdb_id!r}."
        )

    target_dir = Path(target_dir) if target_dir is not None else PDB_DIR
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / f"{pdb_id}.pdb"

    if target.exists():
        return target

    pdbl = PDBList(verbose=False)
    # retrieve_pdb_file writes to pdir as 'pdb{id_lower}.ent' (or .cif).
    try:
        raw = pdbl.retrieve_pdb_file(
            pdb_id,
            pdir=str(target_dir),
            file_format="pdb",
            overwrite=False,
        )
    except Exception as exc:  # network failure, RCSB downtime, ...
        raise RuntimeError(
            f"could not download PDB entry {pdb_id!r} from RCSB: {exc}"
        ) from exc

    raw = Path(raw)
    if not raw.exists():
        # Biopython returns a path even when the entry does not exist, so
        # a missing file here usually means a bad identifier.
        raise RuntimeError(
            f"PDB entry {pdb_id!r} could not be retrieved. Check that the "
            "identifier exists at https://www.rcsb.org and that the entry is "
            "available in legacy PDB format (very large structures are not)."
        )

    raw.rename(target)
    return target


def extract_ca_coordinates(pdb_path, pdb_id=None):
    """Return an ``(n, 3)`` array of C-alpha coordinates from a PDB file.

    Only the first model is used: most PDB files have one, and NMR
    ensembles have several, of which the first is kept. All chains are
    included, so a multi-chain protein like haemoglobin yields the union
    of C-alpha atoms across subunits. Non-standard residues and HETATM
    records (waters, ligands) are skipped.

    Parameters
    ----------
    pdb_path : path-like
        Path to a ``.pdb`` file.
    pdb_id : str, optional
        Identifier used to label the parsed structure. Defaults to the
        file stem.

    Returns
    -------
    coords : ndarray of shape (n, 3)

    Raises
    ------
    ImportError
        If Biopython is not installed.
    FileNotFoundError
        If ``pdb_path`` does not exist.
    ValueError
        If the file cannot be parsed, or holds no C-alpha atoms.
    """
    _, PDBParser = _require_biopython()

    pdb_path = Path(pdb_path)
    if not pdb_path.is_file():
        raise FileNotFoundError(
            f"no PDB file at {pdb_path}. Download it first with "
            "download_pdb(pdb_id)."
        )

    if pdb_id is None:
        pdb_id = pdb_path.stem

    parser = PDBParser(QUIET=True)
    try:
        structure = parser.get_structure(pdb_id, str(pdb_path))
    except Exception as exc:
        raise ValueError(
            f"could not parse {pdb_path} as a PDB file: {exc}"
        ) from exc

    try:
        model = next(structure.get_models())  # first model only
    except StopIteration:
        raise ValueError(
            f"{pdb_path} holds no models; the file is empty or truncated."
        ) from None

    coords = []
    for chain in model:
        for residue in chain:
            # Skip waters, ligands, etc. Standard residues have
            # hetflag == ' '.
            if residue.id[0] != " ":
                continue
            if "CA" in residue:
                coords.append(residue["CA"].get_coord())

    if not coords:
        raise ValueError(
            f"no C-alpha atoms found in {pdb_path}. The file may hold only "
            "HETATM records, or be a nucleic acid structure rather than a "
            "protein."
        )

    return np.array(coords, dtype=np.float64)


def main():
    """Download the three case-study proteins and cache their coordinates."""
    PDB_DIR.mkdir(parents=True, exist_ok=True)
    COORDS_DIR.mkdir(parents=True, exist_ok=True)

    for protein in PROTEINS:
        pdb_id = protein["id"]
        print(f"\n[{pdb_id}] {protein['name']}")

        pdb_path = download_pdb(pdb_id, PDB_DIR)
        print(f"  PDB file: {pdb_path}")

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
