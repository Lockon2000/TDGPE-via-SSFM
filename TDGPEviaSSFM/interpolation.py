"""
This module provides a function, which interpolates data points, giving an EXACT fit. It uses the newton interpolation
method.
"""

from operator import mul
from functools import reduce

import numpy as np


def prod(iterable):
    return reduce(mul, iterable, 1)

def newton_interpol(x_array, y_array):
    # Check that x_array and y_array are the same size and store the size of x_array
    if (n:=x_array.size) != y_array.size:
        raise ValueError("x-array and y-array are not the same size")

    A = np.zeros((n,n))

    # Calculation of The Coefficients Matrix #

    # Initialize the first column of the matrix
    for i in range(n):
        A[i,0] = y_array[i]
    # Calculate the remaining entries
    for j in range(1, n):
        for i in range(n-j):
            A[i,j] = (A[i+1,j-1] - A[i,j-1])/(x_array[i+j] - x_array[i])
    
    # End --- Calculation of The Coefficients Matrix #

    # Construct the interpolation polynom
    def interpolation_polynom(x):
        total = A[0,0]
        for j in range(1, n):
            total += A[0,j]*prod(x - x_array[i] for i in range(j))
        
        return total
    
    return interpolation_polynom

