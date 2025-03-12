#---------------------------------------------------------------------------------
# CCA algorithm implemented in Allo
#---------------------------------------------------------------------------------
# CCA_allo.py
# Canonical Correlation Analysis (CCA) for EEG data

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float64
import allo.ir.types as T

from BCI.Algorithm_kernel.kernel.covariance_kernel import covariance
from BCI.Algorithm_kernel.kernel.pinverse_kernel import pinverse
from BCI.Algorithm_kernel.kernel.eigenvalue_kernel import eigenvalue

def CCA_allo[T: (float64), N: int32, M1: int32, M2: int32](
    X: "T[N, M1]",    # EEG signal data (num_samples, num_channels)
    Y: "T[N, M2]",    # Reference signal data (num_samples, num_harmonics)
    r: "T"            # Maximum correlation coefficient
):
    """
    Implementation of CCA algorithm using Allo
    
    Parameters:
    X -- EEG signal data, shape: (N, M1) -> (1000, 9)
    Y -- Reference signal data, shape: (N, M2) -> (1000, 10)
    r -- Maximum canonical correlation coefficient
    """
    
    # Initialize kernels
    covariance_kernel = covariance(T, M1, M2, N)
    pinverse_kernel = pinverse(T, N)
    eigenvalue_kernel = eigenvalue(T, N)
    
    # Initialize temporary arrays for means
    X_mean: T[M1] = 0.0
    Y_mean: T[M2] = 0.0
    
    # Initialize covariance matrices
    Cxx: T[M1, M1] = 0.0  # Covariance of X
    Cyy: T[M2, M2] = 0.0  # Covariance of Y
    Cxy: T[M1, M2] = 0.0  # Cross-covariance of X and Y
    Cyx: T[M2, M1] = 0.0  # Transpose of Cxy
    
    # Initialize matrices for inverse calculation
    Cxx_inv: T[M1, M1] = 0.0
    Cyy_inv: T[M2, M2] = 0.0
    
    # Temporary matrices for matrix operations
    temp1_M1: T[M1, M1] = 0.0
    temp2_M1: T[M1, M1] = 0.0
    temp1_M2: T[M2, M2] = 0.0
    temp2_M2: T[M2, M2] = 0.0
    
    # Matrix M for eigenvalue calculation
    M: T[M1, M1] = 0.0
    
    # Arrays for eigenvalue calculation
    eigenvals: T[M1] = 0.0
    Q: T[M1, M1] = 0.0
    R: T[M1, M1] = 0.0
    
    # Calculate covariance matrices using covariance kernel
    covariance_kernel(X, X, X_mean, X_mean, Cxx)  # Cxx
    covariance_kernel(Y, Y, Y_mean, Y_mean, Cyy)  # Cyy
    covariance_kernel(X, Y, X_mean, Y_mean, Cxy)  # Cxy
    
    # Transpose Cxy to get Cyx
    for i, j in allo.grid(M2, M1):
        Cyx[i, j] = Cxy[j, i]
    
    # Calculate inverse matrices using inverse kernel
    pinverse_kernel(Cxx, Cxx_inv, temp1_M1, temp2_M1)  # Cxx_inv
    pinverse_kernel(Cyy, Cyy_inv, temp1_M2, temp2_M2)  # Cyy_inv
    
    # Construct matrix M = Cxx_inv @ Cxy @ Cyy_inv @ Cyx
    # Step 1: temp1_M1 = Cxx_inv @ Cxy
    for i, j in allo.grid(M1, M2):
        sum: T = 0.0
        for k in allo.grid(M1):
            sum += Cxx_inv[i, k] * Cxy[k, j]
        temp1_M1[i, j] = sum
    
    # Step 2: temp2_M1 = temp1_M1 @ Cyy_inv
    for i, j in allo.grid(M1, M2):
        sum: T = 0.0
        for k in allo.grid(M2):
            sum += temp1_M1[i, k] * Cyy_inv[k, j]
        temp2_M1[i, j] = sum
    
    # Step 3: M = temp2_M1 @ Cyx
    for i, j in allo.grid(M1, M1):
        sum: T = 0.0
        for k in allo.grid(M2):
            sum += temp2_M1[i, k] * Cyx[k, j]
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

    s = allo.customize(CCA_allo, instantiate=[T, N, M1, M2])
    mod = s.build()
    return mod