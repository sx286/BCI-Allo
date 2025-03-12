# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import numpy as np
import allo
import allo.ir.types as T
from allo.ir.types import float32, float64, int32

def transpose_np(A, A_T, M, N):
    """
    Calculate transpose of matrix A using numpy implementation
    
    Parameters:
    A -- Input matrix, shape: (N, M)
    A_T -- Output transposed matrix, shape: (M, N)
    M -- Number of columns in A
    N -- Number of rows in A
    """
    # Use numpy's built-in function to calculate transpose
    A_T[:] = A.T

def transpose(type, m, n):
    """
    Allo implementation of matrix transpose
    
    Matrix A is a 2D array (N, M)
    Matrix elements are float64 or float32
    M is the number of columns
    N is the number of rows
    """
    def kernel_transpose[T: (float64, float32), M: int32, N: int32](
        A: "T[N, M]",      # Input matrix
        A_T: "T[M, N]"     # Output transposed matrix
    ):
        # Compute transpose
        for i, j in allo.grid(M, N):
            A_T[i, j] = A[j, i]

    s = allo.customize(kernel_transpose, instantiate=[type, m, n])
    mod = s.build()
    return mod

def test_transpose():
    """
    Test transpose implementation against numpy's direct matrix operations
    """
    # Read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    
    # For CI test we use small problem size
    test_psize = "medium"
    M = psize["transpose"][test_psize]["M"]  # Number of columns
    N = psize["transpose"][test_psize]["N"]  # Number of rows
    
    # Generate test data
    A = np.random.randint(-10, 10, (N, M)).astype(np.float64)
    
    # Initialize arrays
    A_T = np.zeros((M, N), dtype=np.float64)
    A_T_ref = np.zeros((M, N), dtype=np.float64)
    
    # Calculate reference result using numpy implementation
    transpose_np(A.copy(), A_T_ref, M, N)
    
    # Calculate result using Allo implementation
    mod = transpose(float64, M, N)
    mod(A, A_T)
    
    # Verify results
    np.testing.assert_allclose(A_T, A_T_ref, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__]) 