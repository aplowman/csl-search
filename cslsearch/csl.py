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
    trial_pair_rotax = np.cross(*trial_pair, axis=0)

    prt(trial_lat, 'trial_lat', t1col)
    prt(trial, 'trial', t1col)
    prt(trial_mag, 'trial_mag', t1)
    prt(trial_mag_idx, 'trial_mag_idx')
    prt(trial_mag_uq, 'trial_mag_uq')
    prt(trial_pair_idx, 'trial_pair_idx', t1row)
    prt(trial_pair, 'trial_pair', t1col_3d)
    prt(trial_pair_rotax, 'trial_pair_rotax', t1col)


if __name__ == '__main__':
    main(int(sys.argv[1]))
