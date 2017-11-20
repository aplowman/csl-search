"""Utility functions."""

import numpy as np


def prt(obj, name):
    """
    Print an object, with it's size and dtype if it's an ndarray.

    Parameters
    ----------
    obj : object
        Object to print.
    name : str
        Name of object to print. 
    """

    if isinstance(obj, np.ndarray):
        print('{} {} {}: \n{}\n'.format(name, obj.shape, obj.dtype, obj))
    else:
        print('{}: \n{}\n'.format(name, obj))
