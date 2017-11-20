"""
Module containing operations on vector-like objects.

"""
from cslsearch import utils
from itertools import permutations
import numpy as np


def find_non_parallel_int_vecs(search_size, dim=3, tile=False):
    """
    Find arbitrary-dimension integer vectors which are non-collinear, whose
    components are less than or equal to a given search size.

    Returned vectors are sorted by magnitude. The zero vector is excluded.

    Parameters
    ----------
    search_size : int
        Positive integer which is the maximum vector component. Memory usage 
        scales quadratically(?) with search_size.
    dim : int
        Dimension of vectors to search.
    tile : bool, optional
        If True, the half-space of dimension `dim` is filled with vectors,
        otherwise just the positive vector components are considered. The
        resulting vector set will still contain only non-collinear vectors.

    Returns
    -------
    ndarray of shape (N, `dim`)

    """

    ss_err_msg = '`search_size` must be an integer greater than zero.'
    if search_size < 1:
        raise ValueError(ss_err_msg)

    try:
        si = np.arange(search_size + 1, dtype=int)
    except TypeError:
        print(ss_err_msg)

    trials = np.vstack(np.meshgrid(*(si,) * dim)).reshape((dim, -1)).T

    # Remove zero vector
    trials = trials[1:]

    sr = si[1:].reshape(-1, 1, 1)
    p = trials * sr
    pv = np.vstack(p)

    # `uinv` indexes `u` to generate the original array pv
    u, uinv = np.unique(pv, axis=0, return_inverse=True)

    # For a given set of (anti-)parallel vectors, we want the smallest
    u_mag = np.sum(u**2, axis=1)
    uinv_mag = u_mag[uinv]
    uinv_mag_rs = np.reshape(uinv_mag, (search_size, -1))
    mag_srt_idx = np.argsort(uinv_mag_rs, axis=0)
    uinv_rs = np.reshape(uinv, (1 * search_size, -1))

    tt = np.tile(np.arange(uinv_rs.shape[1]), (search_size, 1))
    uinv_rs_srt = uinv_rs[mag_srt_idx, tt]

    idx = np.setdiff1d(uinv_rs_srt[0], uinv_rs_srt[1:])

    final_mags = u_mag[idx]
    final_mags_idx = np.argsort(final_mags)

    ret = u[idx][final_mags_idx]

    if tile:

        # Don't want to repeat vectors like [1, 0, 0], [0, 1, 0], [0, 0, 1],
        # so skip the first `dim` vectors:
        tile_base = ret[dim:]

        # For tiling, there will a total of 2^(dim-1) permutations of the
        # original vector set. `dim` - 1 since we want to fill a half space.
        i = np.ones(dim - 1, dtype=int)
        t = np.triu(i, k=1) + -1 * np.tril(i)

        perms_partial = np.array(list(
            set([j for i in t for j in list(permutations(i))])))
        perms = np.hstack([
            np.ones((2**(dim - 1) - 1, 1), dtype=int), perms_partial])
        perms_rs = perms[:, np.newaxis]
        tiled = tile_base * perms_rs

        # Reshape to maintain order by magnitude
        tiled_all = np.vstack([tile_base[np.newaxis], tiled])
        tiled_all_rs = np.vstack(np.swapaxes(tiled_all, 0, 1))
        ret = np.vstack([ret[:dim + 1], tiled_all_rs])

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
