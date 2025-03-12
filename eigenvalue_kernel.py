#----------------------------------------------------------------
# Calculate eigenvalues of a matrix using QR algorithm
#----------------------------------------------------------------

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import numpy as np
import allo
import allo.ir.types as T
from allo.ir.types import float32, float64, int32

#----------------------------------------------------------------
# Numpy implementation of eigenvalue calculation
#----------------------------------------------------------------
def eigenvalue_np(A, eigenvals, N):
    """
    Calculate eigenvalues of matrix A using numpy implementation
    
    Parameters:
    A -- Input matrix, shape: (N, N)
    eigenvals -- Output eigenvalue array, shape: (N,)
    N -- Matrix dimension
    """
    # Use numpy's built-in function to calculate eigenvalues
    eigenvals[:] = np.linalg.eigvals(A)

#----------------------------------------------------------------
# Allo implementation of eigenvalue calculation using QR algorithm
#----------------------------------------------------------------
def eigenvalue(type, n):
    """
    Allo implementation of eigenvalue calculation using QR algorithm
    
    Matrix A should be a square matrix (N, N)
    Matrix elements are float64 or float32
    N is the matrix dimension
    """
    def kernel_eigenvalue[T: (float64, float32), N: int32](
        A: "T[N, N]",          # Input matrix
        eigenvals: "T[N]",     # Output eigenvalues
        Q: "T[N, N]",         # Temporary matrix for Q
        R: "T[N, N]"          # Temporary matrix for R
    ):
        max_iter: int32 = 100  # Maximum number of iterations
        epsilon: T = 1e-10     # Convergence threshold
        
        # Copy input matrix to avoid modifying it
        for i, j in allo.grid(N, N):
            R[i, j] = A[i, j]
        
        # Main QR iteration loop
        for iter in range(max_iter):
            # QR decomposition using Gram-Schmidt process
            for j in allo.grid(N):
                # Copy column j of R to Q
                for i in allo.grid(N):
                    Q[i, j] = R[i, j]
                
                # Orthogonalize against previous vectors
                for k in range(j):
                    dot_product: T = 0.0
                    for i in allo.grid(N):
                        dot_product += Q[i, j] * Q[i, k]
                    
                    for i in allo.grid(N):
                        Q[i, j] -= dot_product * Q[i, k]
                
                # Normalize
                norm: T = 0.0
                for i in allo.grid(N):
                    norm += Q[i, j] * Q[i, j]
                norm = (norm ** 0.5) + epsilon
                
                for i in allo.grid(N):
                    Q[i, j] /= norm
            
            # R = Q^T * A
            for i, j in allo.grid(N, N):
                sum: T = 0.0
                for k in allo.grid(N):
                    sum += Q[k, i] * R[k, j]
                A[i, j] = sum
            
            # A = R * Q for next iteration
            for i, j in allo.grid(N, N):
                sum: T = 0.0
                for k in allo.grid(N):
                    sum += A[i, k] * Q[k, j]
                R[i, j] = sum
        
        # Extract eigenvalues from the diagonal
        for i in allo.grid(N):
            eigenvals[i] = R[i, i]

    s = allo.customize(kernel_eigenvalue, instantiate=[type, n])
    mod = s.build()
    return mod

#----------------------------------------------------------------
# Test eigenvalue implementation against numpy's eigvals
#----------------------------------------------------------------
def test_eigenvalue():
    """
    Test eigenvalue implementation against numpy's eigvals
    """
    # Read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    
    # Test small size first
    test_size = "small"
    N = psize["eigenvalue"][test_size]["N"]
    
    # Generate test matrix (symmetric for real eigenvalues)
    A = np.random.randn(N, N).astype(np.float64)
    A = A @ A.T  # Make symmetric
    
    # Initialize arrays
    eigenvals = np.zeros(N, dtype=np.float64)
    eigenvals_ref = np.zeros(N, dtype=np.float64)
    Q = np.zeros((N, N), dtype=np.float64)
    R = np.zeros((N, N), dtype=np.float64)
    
    # Calculate reference result using numpy
    eigenvalue_np(A.copy(), eigenvals_ref, N)
    
    # Sort reference eigenvalues (absolute values)
    eigenvals_ref = np.sort(np.abs(eigenvals_ref))
    
    # Calculate result using Allo
    mod = eigenvalue(float64, N)
    mod(A, eigenvals, Q, R)
    
    # Sort calculated eigenvalues (absolute values)
    eigenvals = np.sort(np.abs(eigenvals))
    
    # Verify results
    try:
        np.testing.assert_allclose(eigenvals, eigenvals_ref, rtol=1e-2, atol=1e-2)
        print(f"✓ {test_size} size test passed")
    except AssertionError as e:
        print(f"✗ {test_size} size test failed:")
        print(f"  Expected: {eigenvals_ref}")
        print(f"  Got: {eigenvals}")
        raise

if __name__ == "__main__":
    pytest.main([__file__]) 