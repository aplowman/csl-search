"""
Module for searching for CSL vectors within a given lattice system.

"""
import sys
import numpy as np
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

    # Generate a set of non-collinear integer vectors within given search size
    trial_lat = vectors.find_non_parallel_int_vecs(search_size, tile=True)
    trial = trial_lat @ lattice
    trial_mag = np.linalg.norm(trial, axis=1)

    # Group trial vectors by equal magnitudes
    trial_mag_idx = [[k] + v for k,
                     v in vectors.get_equal_indices(trial_mag)[0].items()]
    trial_mag_uq = trial_mag[[i[0] for i in trial_mag_idx]]

    prt(trial_lat, 'trial_lat')
    prt(trial, 'trial')
    prt(trial_mag, 'trial_mag')
    prt(trial_mag_idx, 'trial_mag_idx')
    prt(trial_mag_uq, 'trial_mag_uq')


if __name__ == '__main__':
    main(int(sys.argv[1]))
