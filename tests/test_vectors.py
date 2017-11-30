"""Module defining unit tests on the cslsearch.vectors module."""

import unittest
from cslsearch import vectors
import numpy as np


class FindNonParallelIntVecsTestCase(unittest.TestCase):
    """Test case for `vectors.find_non_parallel_int_vecs`."""

    def test_search_size_components(self):
        """Check the maximum vector component is equal to the search size."""
        search_size = 10
        v = vectors.find_non_parallel_int_vecs(search_size, tile=False)
        self.assertEqual(np.max(v), search_size)

    def test_invalid_search_size(self):
        """Test exceptions are raised for invalid search sizes."""
        invalid_ss = range(-2, 1)
        for i in invalid_ss:
            with self.assertRaises(ValueError):
                vectors.find_non_parallel_int_vecs(i)

    def test_non_parallel_tiled(self):
        """
        Test for a search size of fifteen with `tile` True, found vectors are 
        non (anti-)parallel.

        """
        v = vectors.find_non_parallel_int_vecs(15, tile=True)

        # Find cross product of each vector with all vectors
        cross_self = np.cross(v, v[:, np.newaxis])
        cross_self_zero = np.all(cross_self == 0, axis=2)
        self.assertEqual(np.max(np.sum(cross_self_zero, axis=1)), 1)

    def test_non_parallel(self):
        """
        Test for a search size of fifteen with `tile` False found vectors are
        non (anti-)parallel.

        """
        v = vectors.find_non_parallel_int_vecs(15, tile=False)

        # Find cross product of each vector with all vectors
        cross_self = np.cross(v, v[:, np.newaxis])
        cross_self_zero = np.all(cross_self == 0, axis=2)
        self.assertEqual(np.max(np.sum(cross_self_zero, axis=1)), 1)


if __name__ == '__main__':
    unittest.main()
