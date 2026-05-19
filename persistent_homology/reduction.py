"""Reduction algorithms for persistent homology.

Three strategies are provided, all computing the same persistence pairs:

* :func:'reduce_standard'; the standard column reduction of
  Zomorodian and Carlsson.
* :func:'reduce_with_clearing'; the same reduction with the clearing
  optimisation, processing dimensions from high to low.
* :func:'reduce_pcoh'; the persistent cohomology algorithm (pCoh) of
  de Silva, Morozov, and Vejdemo-Johansson.

"""

from collections import Counter, defaultdict


def add_columns_z2(col_a, col_b):
    """Symmetric difference of two sorted lists.

    Computes "col_a + col_b" over Z/2Z, where each column is a sorted
    list of row indices. Entries appearing in both columns cancel, since
    "1 + 1 = 0" modulo 2.

    Parameters
    ----------
    col_a, col_b : list of int
        Sorted lists of row indices.

    Returns
    -------
    result : list of int
        The sorted symmetric difference.
    """
    result = []
    i, j = 0, 0
    while i < len(col_a) and j < len(col_b):
        if col_a[i] < col_b[j]:
            result.append(col_a[i])
            i += 1
        elif col_a[i] > col_b[j]:
            result.append(col_b[j])
            j += 1
        else:  # equal: 1 + 1 = 0 mod 2
            i += 1
            j += 1
    result.extend(col_a[i:])
    result.extend(col_b[j:])
    return result


def reduce_standard(columns):
    """Standard persistence algorithm.

    Parameters
    ----------
    columns : list of lists
        The boundary matrix in column-wise sparse form.

    Returns
    -------
    reduced : list of lists
        The reduced columns.
    pairs : list of (birth_idx, death_idx)
        The persistence pairs, as filtration indices.

    """
    m = len(columns)
    reduced = [col[:] for col in columns]
    pivot_col = {}
    pairs = []

    for j in range(m):
        while (reduced[j] and
               reduced[j][-1] in pivot_col):
            i = pivot_col[reduced[j][-1]]
            reduced[j] = add_columns_z2(reduced[j], reduced[i])
        if reduced[j]:
            pivot = reduced[j][-1]
            pivot_col[pivot] = j
            pairs.append((pivot, j))

    return reduced, pairs


def reduce_with_clearing(columns, filtration):
    """Persistence with the clearing optimisation.

    Processes dimensions from high to low. After reducing dimension
    "k+1", any "(k+1)"-pivot tells us that the simplex at that pivot
    index will reduce to zero (Clearing Lemma), so its reduction at
    dimension "k" is skipped.

    Parameters
    ----------
    columns : list of lists
        The boundary matrix in column-wise sparse form.
    filtration : list of (filtration_value, simplex_tuple)
        The filtration, used to group columns by simplex dimension.

    Returns
    -------
    reduced : list of lists
        The reduced columns.
    pairs : list of (birth_idx, death_idx)
        The persistence pairs, as filtration indices.
    """
    reduced = [col[:] for col in columns]
    pivot_col = {}      # pivot row -> column index
    is_birth = set()    # indices known to be births
    pairs = []

    # Group column indices by simplex dimension
    dim_groups = {}
    for j, (_, simplex) in enumerate(filtration):
        d = len(simplex) - 1
        dim_groups.setdefault(d, []).append(j)

    if not dim_groups:
        return reduced, pairs
    max_dim = max(dim_groups.keys())

    # Process from highest to lowest dimension
    for k in range(max_dim, -1, -1):
        if k not in dim_groups:
            continue
        for j in dim_groups[k]:
            if j in is_birth:
                reduced[j] = []   # cleared: skip reduction
                continue
            # Standard reduction step
            while (reduced[j] and
                   reduced[j][-1] in pivot_col):
                i = pivot_col[reduced[j][-1]]
                reduced[j] = add_columns_z2(
                    reduced[j], reduced[i])
            if reduced[j]:
                pivot = reduced[j][-1]
                pivot_col[pivot] = j
                # Mark pivot row as a birth simplex (its column will be cleared at next dim)
                is_birth.add(pivot)
                pairs.append((pivot, j))

    return reduced, pairs


def reduce_pcoh(filtration):
    """Persistent cohomology algorithm (pCoh).

    Implementation of Algorithm 3 of de Silva, Morozov, and
    Vejdemo-Johansson (2011), specialised to Z/2Z coefficients. The
    filtration is traversed in order while a set of live cocycles is
    maintained; each simplex either creates a new cocycle (a birth) or
    kills the youngest candidate cocycle (a death).

    Parameters
    ----------
    filtration : list of (filtration_value, simplex_tuple)
        The filtration, as returned by :func:'vr_filtration'.

    Returns
    -------
    pairs : list of (birth_idx, death_idx)
        The persistence pairs, as filtration indices.
    """
    active = {}    # cocycle_id -> set of simplex indices (the support)
    births = {}    # cocycle_id -> filtration index of birth simplex
    owners = defaultdict(set)  # simplex_idx -> set of cocycle_ids containing it
    next_id = 0
    pairs = []

    simplex_idx = {tuple(s): i for i, (_, s) in enumerate(filtration)}

    for j, (_, sigma) in enumerate(filtration):
        # Find candidates: cocycles z such that sigma is in delta(z),
        # i.e. cocycles containing an odd number of faces of sigma.
        if len(sigma) <= 1:
            candidates = []
        else:
            face_indices = [simplex_idx[sigma[:i] + sigma[i + 1:]]
                            for i in range(len(sigma))]
            count = Counter()
            for tau_idx in face_indices:
                for cid in owners[tau_idx]:
                    count[cid] += 1
            candidates = [cid for cid, c in count.items() if c % 2 == 1]

        if not candidates:
            # New cocycle is born
            cid = next_id
            next_id += 1
            active[cid] = {j}
            births[cid] = j
            owners[j].add(cid)
        else:
            # Kill the youngest candidate (largest birth index).
            pivot = max(candidates, key=lambda c: births[c])
            pairs.append((births[pivot], j))

            # Add the pivot's support to all other candidates
            pivot_support = active[pivot]
            for cid in candidates:
                if cid == pivot:
                    continue
                added = pivot_support - active[cid]
                removed = pivot_support & active[cid]
                active[cid] = active[cid] ^ pivot_support
                for s in added:
                    owners[s].add(cid)
                for s in removed:
                    owners[s].discard(cid)

            # Drop the pivot cocycle
            for s in pivot_support:
                owners[s].discard(pivot)
            del active[pivot]
            del births[pivot]

    return pairs
