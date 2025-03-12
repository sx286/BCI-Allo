# cca.py
# Canonical Correlation Analysis (CCA) for EEG data using Numpy

import os
import json
import pytest
import allo
import numpy as np
from allo.ir.types import int32, float64
import allo.ir.types as T

#---------------------------------------------------------------------------------
# CCA algorithm implemented in numpy
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
    
    # Calculate covariance matrices
    n = X.shape[0]  # number of samples
    Cxx = np.dot(X.T, X) / (n-1)  # covariance matrix of X: X^T * X / (n-1)
    Cyy = np.dot(Y.T, Y) / (n-1)  # covariance matrix of Y: Y^T * Y / (n-1)
    Cxy = np.dot(X.T, Y) / (n-1)  # cross-covariance matrix of X and Y: X^T * Y / (n-1)
    
    try:
        # 3. Calculate inverse matrices
        Cxx_inv = np.linalg.inv(Cxx)
        Cyy_inv = np.linalg.inv(Cyy)
        
        # 4. Construct matrix for eigenvalue problem
        # M = Cxx^(-1) * Cxy * Cyy^(-1) * Cxy.T
        M = Cxx_inv @ Cxy @ Cyy_inv @ Cxy.T
        
        # 5. Calculate eigenvalues
        eigenvalues = np.linalg.eigvals(M)
        
        # 6. Square root of maximum eigenvalue is the correlation coefficient
        r = np.sqrt(np.abs(np.max(eigenvalues)))
        
    except np.linalg.LinAlgError:
        # Return 0 if matrix is not invertible
        r = 0.0
        
    return r

