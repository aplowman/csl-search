"""
Module for searching for CSL vectors within a given lattice system.

"""
import sys
import numpy as np
from itertools import combinations
from cslsearch.utils import prt
from cslsearch import vectors


def main(search_size):
    """
    Function that performs main algorithm.

    Parameters
    ----------
    search_size : int

    Returns
    -------
    ndarray of shape (3, N)

    Notes
    -----
    Column vectors operated on by pre-multiplication are mostly used.

    TODO:
        - Improve get_equal_indices. Can we simplify return?

    """

    # Hard-coded for now:
    lat_a = 3.2316
    lat_c = 5.1475
    lattice = np.array([
        [lat_a, -lat_a / 2, 0],
        [0, lat_a * np.cos(np.radians(30)), 0],
        [0, 0, lat_c],
    ])
    prt(lattice, 'lattice')

    # DEBUGGING
    # ---------
    # Print the first 6 (inner) column/row vectors
    t1 = (slice(0, 5),)
    t1col = (slice(None),) + t1
    t1row = t1 + (slice(None),)
    t1col_3d = (slice(None), slice(None)) + t1
    # ---------
    # DEBUGGING

    # Generate a set of non-collinear integer vectors within given search size
    trial_lat = vectors.find_non_parallel_int_vecs(search_size, tile=True)
    trial_lat = trial_lat.T
    trial = lattice @ trial_lat
    trial_mag = np.linalg.norm(trial, axis=0)

    # Group trial vectors by equal magnitudes
    trial_mag_idx = [[k] + v for k,
                     v in vectors.get_equal_indices(trial_mag)[0].items()]
    trial_mag_uq_idx = [i[0] for i in trial_mag_idx]
    trial_mag_uq = trial_mag[trial_mag_uq_idx]

    # Generate vector pair indices which correspond to equal magnitude vectors
    trial_pair_idx = np.array([i for x in trial_mag_idx
                               for i in list(combinations(x, 2))])

    # Find rotation axes for each trial pair
    trial_pair = trial[:, trial_pair_idx].transpose(2, 0, 1)
    tp_rotax = np.cross(*trial_pair, axis=0)
    tp_rotax_u = tp_rotax / np.linalg.norm(tp_rotax, axis=0)
    tp_dot = np.einsum('ij,ij->j', *trial_pair)

    prt(trial_lat, 'trial_lat', t1col)
    prt(trial, 'trial', t1col)
    prt(trial_mag, 'trial_mag', t1)
    prt(trial_mag_idx, 'trial_mag_idx')
    prt(trial_mag_uq, 'trial_mag_uq')
    prt(trial_pair_idx, 'trial_pair_idx', t1row)
    prt(trial_pair, 'trial_pair', t1col_3d)
    prt(tp_rotax, 'tp_rotax', t1col)
    prt(tp_rotax_u, 'tp_rotax_u', t1col)
    prt(tp_dot, 'tp_dot', t1)

    # Find which trial pair rotation axes are vectors within the search size.
    # `rotax_idx`: keys index `trial`, values are lists which index `trial_pair`
    rotax_idx = vectors.get_parallel_idx(trial, tp_rotax)

    print('Found {} rotation axes within search size.'.format(len(rotax_idx)))
    prt(rotax_idx, 'rotax_idx')

    # Loop through each rotation axis:
    for rotax_idx_i, trial_pair_idx_i in rotax_idx.items():

        prt(rotax_idx_i, 'rotax_idx_i')
        prt(trial_pair_idx_i, 'trial_pair_idx_i')

        # Find rotation angle for each vector pair about this axis:
        rotax_i = trial[:, rotax_idx_i][:, np.newaxis]
        rotax_unit_i = rotax_i / np.linalg.norm(rotax_i)
        trial_pair_i = trial_pair[:, :, trial_pair_idx_i]  # Not needed

        prt(rotax_i, 'rotax_i')
        prt(rotax_unit_i, 'rotax_unit_i')
        prt(trial_pair_i, 'trial_pair_i')

        # Get cross and dot products (previously computed):
        tp_i_cross = tp_rotax[:, trial_pair_idx_i]
        tp_i_dot = tp_dot[trial_pair_idx_i]

        prt(tp_i_cross, 'tp_i_cross')
        prt(tp_i_dot, 'tp_i_dot')

        # Get (signed) rotation angles about this axis:
        atan_arg1 = np.einsum('ij,ik->k', rotax_unit_i, tp_i_cross)
        atan_arg2 = tp_i_dot
        theta_i = np.arctan2(atan_arg1, atan_arg2)
        theta_i_deg = np.rad2deg(theta_i)

        prt(atan_arg1, 'atan_arg1')
        prt(atan_arg2, 'atan_arg2')
        prt(theta_i, 'theta_i')
        prt(theta_i_deg, 'theta_i_deg')

        break


if __name__ == '__main__':
    main(int(sys.argv[1]))
