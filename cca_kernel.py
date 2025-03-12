#----------------------------------------------------------------
# Compare CCA implementations between numpy and allo
#----------------------------------------------------------------

# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import numpy as np
import allo
import allo.ir.types as T
from allo.ir.types import float32, float64, int32, uint8, uint16

#---------------------------------------------------------------------------------
# CCA algorithm using numpy
#---------------------------------------------------------------------------------

def CCA_np(X: np.ndarray, Y: np.ndarray):
    """
    Implementation of CCA algorithm
    
    Parameters:
    X -- EEG signal data, shape: (num_samples, num_channels)
                                      1000           9 
    Y -- Reference signal data, shape: (num_samples, num_harmonics)
                                            1000           10
    Returns:
    r -- Maximum canonical correlation coefficient

    In CCA, we need to find a linear combination of X and Y that maximizes the correlation between them.
    V = a'X  X: (1000, 9 )   a: (9, 1)  - X's projection vector -> (1000, 1)
    W = b'Y  Y: (1000, 10)   b: (10, 1) - Y's projection vector -> (1000, 1)
    """
    
    # Ensure input type is float64
    X = X.astype(np.float64)
    Y = Y.astype(np.float64)
    
    # 1. Calculate covariance matrices
    # Cov(X,X) = E[(X - μ)(X - μ)ᵀ]    covariance matrix of X
    # Cov(X,Y) = E[(X - μx)(Y - μy)ᵀ]  cross-covariance matrix of X and Y
    
    # Center the data
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    
    # # Calculate covariance matrices
    n = X.shape[0]  # number of samples
    Cxx = np.dot(X.T, X) / (n-1)  # covariance matrix of X: X^T * X / (n-1) shape: (9, 9)
    Cyy = np.dot(Y.T, Y) / (n-1)  # covariance matrix of Y: Y^T * Y / (n-1) shape: (10, 10)
    Cxy = np.dot(X.T, Y) / (n-1)  # cross-covariance matrix of X and Y: X^T * Y / (n-1) shape: (9, 10)
    
    # Use pseudo-inverse instead of normal inverse matrix
    Cxx_inv = np.linalg.pinv(Cxx)
    Cyy_inv = np.linalg.pinv(Cyy)
    
    # Construct matrix for eigenvalue problem
    # shape: (9, 9) @ (9, 10) @ (10, 10) @ (10, 9) -> (9, 9)
    M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
    
    # Calculate eigenvalues
    eigenvalues = np.linalg.eigvals(M)
    
    # Square root of maximum eigenvalue is the correlation coefficient
    r = np.sqrt(np.abs(np.max(eigenvalues)))

    return r

#----------------------------------------------------------------
# Allo implementation of CCA algorithm
#----------------------------------------------------------------

from BCI.Algorithm_kernel.kernel.covariance_kernel import covariance
from BCI.Algorithm_kernel.kernel.pinverse_kernel import pinverse
from BCI.Algorithm_kernel.kernel.eigenvalue_kernel import eigenvalue

def CCA_Allo(type, n, m1, m2):
    """
    Allo implementation of CCA algorithm
    
    Parameters:
    type -- Data type (float64)
    n -- Number of samples (N=1000)
    m1 -- Number of EEG channels (M1=9)
    m2 -- Number of reference harmonics (M2=10)
    """
    def kernel_cca[T: (float64, float32), N: int32, M1: int32, M2: int32](
        X: "T[N, M1]",    # EEG signal data (1000, 9)
        Y: "T[N, M2]",    # Reference signal data (1000, 10)
        r: "T"            # Output correlation coefficient
    ):
        # Initialize kernels
        covariance_kernel = covariance(T, M1, M2, N)
        pinverse_kernel = pinverse(T, M1)  # For Cxx_inv (9x9)
        pinverse_kernel2 = pinverse(T, M2)  # For Cyy_inv (10x10)
        eigenvalue_kernel = eigenvalue(T, M1)  # For M (9x9)
        
        # Initialize temporary arrays for means
        X_mean: T[M1] = 0.0  # Mean of X channels
        Y_mean: T[M2] = 0.0  # Mean of Y harmonics
        
        # Initialize covariance matrices
        Cxx: T[M1, M1] = 0.0  # Covariance of X (9x9)
        Cyy: T[M2, M2] = 0.0  # Covariance of Y (10x10)
        Cxy: T[M1, M2] = 0.0  # Cross-covariance of X and Y (9x10)
        Cyx: T[M2, M1] = 0.0  # Transpose of Cxy (10x9)
        
        # Initialize matrices for inverse calculation
        Cxx_inv: T[M1, M1] = 0.0  # (9x9)
        Cyy_inv: T[M2, M2] = 0.0  # (10x10)
        
        # Temporary matrices for matrix operations
        temp1: T[M1, M2] = 0.0  # (9x10)
        temp2: T[M1, M2] = 0.0  # (9x10)
        
        # Matrix M for eigenvalue calculation (9x9)
        M: T[M1, M1] = 0.0
        
        # Arrays for eigenvalue calculation
        eigenvals: T[M1] = 0.0  # 9 eigenvalues
        Q: T[M1, M1] = 0.0
        R: T[M1, M1] = 0.0
        
        # Calculate covariance matrices using covariance kernel
        covariance_kernel(X, X, X_mean, X_mean, Cxx)  # Cxx (9x9)
        covariance_kernel(Y, Y, Y_mean, Y_mean, Cyy)  # Cyy (10x10)
        covariance_kernel(X, Y, X_mean, Y_mean, Cxy)  # Cxy (9x10)
        
        # Transpose Cxy to get Cyx
        for i, j in allo.grid(M2, M1):
            Cyx[i, j] = Cxy[j, i]
        
        # Calculate inverse matrices using inverse kernel
        pinverse_kernel(Cxx, Cxx_inv, Q, R)  # Cxx_inv (9x9)
        pinverse_kernel2(Cyy, Cyy_inv, Q, R)  # Cyy_inv (10x10)
        
        # Construct matrix M = Cxx_inv @ Cxy @ Cyy_inv @ Cyx
        # Step 1: temp1 = Cxx_inv @ Cxy (9x9 @ 9x10 -> 9x10)
        for i, j in allo.grid(M1, M2):
            sum: T = 0.0
            for k in allo.grid(M1):
                sum += Cxx_inv[i, k] * Cxy[k, j]
            temp1[i, j] = sum
        
        # Step 2: temp2 = temp1 @ Cyy_inv (9x10 @ 10x10 -> 9x10)
        for i, j in allo.grid(M1, M2):
            sum: T = 0.0
            for k in allo.grid(M2):
                sum += temp1[i, k] * Cyy_inv[k, j]
            temp2[i, j] = sum
        
        # Step 3: M = temp2 @ Cyx (9x10 @ 10x9 -> 9x9)
        for i, j in allo.grid(M1, M1):
            sum: T = 0.0
            for k in allo.grid(M2):
                sum += temp2[i, k] * Cyx[k, j]
            M[i, j] = sum
        
        # Calculate eigenvalues using eigenvalue kernel
        eigenvalue_kernel(M, eigenvals, Q, R)
        
        # Find maximum eigenvalue
        max_eigenval: T = 0.0
        for i in allo.grid(M1):
            val: T = eigenvals[i] if eigenvals[i] >= 0.0 else -eigenvals[i]
            if val > max_eigenval:
                max_eigenval = val
        
        # Set correlation coefficient as square root of maximum eigenvalue
        r = (max_eigenval ** 0.5)

    s = allo.customize(kernel_cca, instantiate=[type, num_samples, num_channels, num_harmonics])
    mod = s.build()
    return mod

#----------------------------------------------------------------
# Test CCA implementation against numpy's implementation
#----------------------------------------------------------------

def test_cca():
    """
    Test CCA implementation against numpy's implementation
    """
    # Use actual data dimensions for testing
    N = 1000    # Number of samples
    M1 = 9      # Number of EEG channels
    M2 = 10     # Number of reference harmonics
    
    # Generate test data
    X = np.random.randint(-10, 10, (N, M1)).astype(np.float64)  # EEG data (1000, 9)
    Y = np.random.randint(-10, 10, (N, M2)).astype(np.float64)  # Reference data (1000, 10)
    
    # Initialize output variables
    r = np.zeros(1, dtype=np.float64)[0]
    r_ref = np.zeros(1, dtype=np.float64)[0]
    
    # Use numpy implementation to calculate reference result
    r_ref = CCA_np(X.copy(), Y.copy())
    
    # Use Allo implementation to calculate result
    mod = CCA_Allo(float64, N, M1, M2)
    mod(X, Y, r)
    
    # Verify results
    print("\nCCA test results:")
    print(f"  - numpy result: {r_ref:.6f}")
    print(f"  - allo result: {r:.6f}")
    print(f"  - absolute error: {abs(r - r_ref):.6f}")
    
    np.testing.assert_allclose(r, r_ref, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
