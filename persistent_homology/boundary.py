"""Boundary matrix construction and Z/2Z column arithmetic.

The boundary matrix encodes, for each simplex, its codimension-1 faces.
Over Z/2Z a column is stored simply as the sorted list of row indices
where it is non-zero, and column addition is symmetric difference.
"""


def build_boundary_matrix(filtration):
    """Build the combined boundary matrix.

    Parameters
    ----------
    filtration : list of (filtration_value, simplex_tuple)
        The filtration, as returned by :func:`vr_filtration`.

    Returns
    -------
    columns : list of lists
        "columns[j]" is the sorted list of row indices "i" such that
        simplex "i" is a codimension-1 face of simplex "j".
    """
    # Map each simplex to its index in the filtration
    simplex_to_idx = {}
    for idx, (_, simplex) in enumerate(filtration):
        simplex_to_idx[simplex] = idx

    m = len(filtration)
    columns = [[] for _ in range(m)]

    for j, (_, simplex) in enumerate(filtration):
        if len(simplex) <= 1:
            continue  # 0-simplices have empty boundary
        # Faces: remove each vertex in turn
        for i in range(len(simplex)):
            face = simplex[:i] + simplex[i + 1:]
            face_idx = simplex_to_idx[face]
            columns[j].append(face_idx)
        columns[j].sort()

    return columns
