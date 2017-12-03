"""
Module containing operations on vector-like objects.

"""
from itertools import permutations, combinations
import numpy as np


def find_positive_int_vecs(search_size, dim=3):
    """
    Find arbitrary-dimension positive integer vectors which are
    non-collinear whose components are less than or equal to a
    given search size. Vectors with zero components are not included.

    Non-collinear here means no two vectors are related by a scaling factor.

    Parameters
    ----------
    search_size : int
        Positive integer which is the maximum vector component
    dim : int
        Dimension of vectors to search.

    Returns
    -------
    ndarray of shape (N, `dim`)

    """

    # Generate trial vectors as a grid of integer vectors
    search_ints = np.arange(1, search_size + 1)
    search_grid = np.meshgrid(*(search_ints,) * dim)
    trials = np.vstack(search_grid).reshape((dim, -1)).T

    # Multiply each trial vector by each possible integer up to
    # `search_size`:
    search_ints_rs = search_ints.reshape(-1, 1, 1)
    trial_combs = trials * search_ints_rs

    # Combine trial vectors and their associated scaled vectors:
    trial_combs_all = np.vstack(trial_combs)

    # Find unique vectors. The inverse indices`uinv` indexes
    # the set of unique vectors`u` to generate the original array `pv`:
    uniq, uniq_inv = np.unique(trial_combs_all, axis=0, return_inverse=True)

    # For a given set of (anti-)parallel vectors, we want the smallest, so get
    # their relative magnitudes. This is neccessary since `np.unique` does not
    # return vectors sorted in a sensible way if there are negative components.
    # (But we do we have negative components here?)
    uniq_mag = np.sum(uniq**2, axis=1)

    # Get the magnitudes of just the directionally-unique vectors:
    uniq_inv_mag = uniq_mag[uniq_inv]

    # Reshape the magnitudes to allow sorting for a given scale factor:
    uniq_inv_mag_rs = np.reshape(uniq_inv_mag, (search_size, -1))

    # Get the indices which sort the trial vectors
    mag_srt_idx = np.argsort(uniq_inv_mag_rs, axis=0)

    # Reshape the inverse indices
    uniq_inv_rs = np.reshape(uniq_inv, (search_size, -1))

    # Sort the inverse indices by their corresponding vector magnitudes,
    # for each scale factor:
    col_idx = np.tile(np.arange(uniq_inv_rs.shape[1]), (search_size, 1))
    uniq_inv_rs_srt = uniq_inv_rs[mag_srt_idx, col_idx]

    # Only keep inverse indices in first row which are not in any other row.
    # First row indexes lowest magnitude vectors for each scale factor.
    idx = np.setdiff1d(uniq_inv_rs_srt[0], uniq_inv_rs_srt[1:])

    # Sort kept vectors by magnitude
    final_mags = uniq_mag[idx]
    final_mags_idx = np.argsort(final_mags)

    ret = uniq[idx][final_mags_idx]

    return ret


def tile_int_vecs(int_vecs, dim):
    """
    Tile arbitrary-dimension integer vectors such that they occupy a
    half-space.

    """
    # For tiling, there will a total of 2^(`dim` - 1) permutations of the
    # original vector set. (`dim` - 1) since we want to fill a half space.
    i = np.ones(dim - 1, dtype=int)
    t = np.triu(i, k=1) + -1 * np.tril(i)

    # Get permutation of +/- 1 factors to tile initial vectors into half-space
    perms_partial_all = [j for i in t for j in list(permutations(i))]
    perms_partial = np.array(list(set(perms_partial_all)))

    perms_first_col = np.ones((2**(dim - 1) - 1, 1), dtype=int)
    perms_first_row = np.ones((1, dim), dtype=int)
    perms_non_eye = np.hstack([perms_first_col, perms_partial])
    perms = np.vstack([perms_first_row, perms_non_eye])

    perms_rs = perms[:, np.newaxis]
    tiled = int_vecs * perms_rs
    ret = np.vstack(tiled)

    return ret


def find_non_parallel_int_vecs(search_size, dim=3, tile=False):
    """
    Find arbitrary-dimension integer vectors which are non-collinear, whose
    components are less than or equal to a given search size.

    Non-collinear here means no two vectors are related by a scaling factor.
    The zero vector is excluded.

    Parameters
    ----------
    search_size : int
        Positive integer which is the maximum vector component.
    dim : int
        Dimension of vectors to search.
    tile : bool, optional
        If True, the half-space of dimension `dim` is filled with vectors,
        otherwise just the positive vector components are considered. The
        resulting vector set will still contain only non-collinear vectors.

    Returns
    -------
    ndarray of shape (N, `dim`)
        Vectors are not globally ordered.

    Notes
    -----
    Searching for vectors with `search_size` of 100 uses about 9 GB of memory.

    """

    # Find all non-parallel positive integer vectors which have no
    # zero components:
    ret = find_positive_int_vecs(search_size, dim)

    # If requested, tile the vectors such that they occupy a half-space:
    if tile and dim > 1:
        ret = tile_int_vecs(ret, dim)

    # Add in the vectors which are contained within a subspace of dimension
    # (`dim` - 1) on the principle axes. I.e. vectors with zero components:
    if dim > 1:

        # Recurse through each (`dim` - 1) dimension subspace:
        low_dim = dim - 1
        vecs_lower = find_non_parallel_int_vecs(search_size, low_dim, tile)

        # Raise vectors to current dimension with a zero component. The first
        # (`dim` - 1) "prinicple" vectors (of the form [1, 0, ...]) should be
        # considered separately, else they will be repeated.
        principle = np.eye(dim, dtype=int)
        non_prcp = vecs_lower[low_dim:]

        if non_prcp.size:

            edges_shape = (dim, non_prcp.shape[0], non_prcp.shape[1] + 1)
            vecs_edges = np.zeros(edges_shape, dtype=int)
            edges_idx = list(combinations(list(range(dim)), low_dim))

            for i in range(dim):
                vecs_edges[i][:, edges_idx[i]] = non_prcp

            vecs_edges = np.vstack([principle, *vecs_edges])

        else:
            vecs_edges = principle

        ret = np.vstack([vecs_edges, ret])

    return ret


def get_parallel_idx(vecs_a, vecs_b):
    """
    Find which vectors in an array of vectors are (anti-)parallel to which
    vectors in another array of vectors, according to their cross product.

    Parameters
    ----------
    vecs_a : ndarray of shape (3, N)
        Array of column vectors
    vecs_b : ndarray of shape (3, M)
        Array of column vectors

    Returns
    -------
    dict of (int: ndarray of int of shape (P,))
        Returned dict has keys which index `vecs_a` and values which are
        integer arrays which index `vecs_b`.

    """

    ret = {}
    for vec_a_idx, vec_a in enumerate(vecs_a.T):

        vec_a = vec_a[:, np.newaxis]
        vec_a_cross = np.cross(vec_a, vecs_b, axis=0)

        is_parallel = np.all(np.isclose(vec_a_cross, 0.0), axis=0)
        parallel_idx = np.where(is_parallel)[0]

        if parallel_idx.size:
            ret.update({
                vec_a_idx: parallel_idx
            })

    return ret


def get_equal_indices(arr, scale_factors=None):
    """
    Return the indices along the first dimension of an array which index equal
    sub-arrays.

    Parameters
    ----------
    arr : ndarray or list
        Array or list of any shape whose elements along its first dimension are
        compared for equality.
    scale_factors : list of float or list of int, optional
        Multiplicative factors to use when checking for equality between
        subarrays. Each factor is checked independently.

    Returns
    -------
    tuple of dict of int: list of int
        Each tuple item corresponds to a scale factor for which each dict maps
        a subarray index to a list of equivalent subarray indices given that
        scale factor. Length of returned tuple is equal to length of
        `scale_factors` or 1 if `scale_factors` is not specified.

    Notes
    -----
    If we have a scale factor `s` which returns {a: [b, c, ...]}, then the
    inverse scale factor `1/s` will return {b: [a], c: [a], ...}.


    Examples
    --------

    1D examples:

    >>> a = np.array([5, 1, 4, 6, 1, 8, 2, 7, 4, 7])
    >>> get_equal_indices(a)
    ({1: [4], 2: [8], 7: [9]},)

    >>> a = np.array([1, -1, -1, 2])
    >>> get_equal_indices(a, scale_factors=[1, -1, -2, -0.5])
    ({1: [2]}, {0: [1, 2]}, {1: [3], 2: [3]}, {3: [1, 2]})

    2D example:

    >>> a = np.array([[1., 2.], [3., 4.], [-0.4, -0.8]])
    >>> get_equal_indices(a, scale_factors=[-0.4])
    ({0: [2]},)

    """

    if not isinstance(arr, np.ndarray):
        arr = np.array(arr)

    if scale_factors is None:
        scale_factors = [1]

    a_dims = len(arr.shape)
    arr_B = arr[:, np.newaxis]

    sf_shape = tuple([len(scale_factors)] + [1] * (a_dims + 1))
    sf = np.array(scale_factors).reshape(sf_shape)

    bc = np.broadcast_arrays(arr, arr_B, sf)
    c = np.isclose(bc[0], bc[1] * bc[2])

    if a_dims > 1:
        c = np.all(c, axis=tuple(range(3, a_dims + 2)))

    out = ()
    for c_sub in c:

        w2 = np.where(c_sub)
        d = {}
        skip_idx = []

        for i in set(w2[0]):

            if i not in skip_idx:

                row_idx = np.where(w2[0] == i)[0]
                same_idx = list(w2[1][row_idx])

                if i in same_idx:

                    if len(row_idx) == 1:
                        continue

                    elif len(row_idx) > 1:
                        same_idx.remove(i)

                d.update({i: same_idx})
                skip_idx += same_idx

        out += (d,)

    return out
