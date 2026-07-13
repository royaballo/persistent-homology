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

    Raises
    ------
    ValueError
        If a simplex has a face that is absent from the filtration. A
        filtration must be closed under taking faces for its boundary
        matrix to be defined.
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
            try:
                face_idx = simplex_to_idx[face]
            except KeyError:
                # A Vietoris-Rips filtration is always closed under faces
                # (a face cannot have a larger diameter than the simplex
                # it bounds), so reaching this point means the filtration
                # was built by other means and is not a valid complex.
                raise ValueError(
                    f"simplex {simplex} has the face {face}, which is absent "
                    "from the filtration. A filtration must be closed under "
                    "taking faces; build it with vr_filtration()."
                ) from None
            columns[j].append(face_idx)
        columns[j].sort()

    return columns
