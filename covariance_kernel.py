#----------------------------------------------------------------
# Calculate covariance between two matrices
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

#----------------------------------------------------------------
# Numpy implementation of covariance calculation
#----------------------------------------------------------------
def covariance_np(data1, data2, mean1, mean2, cov, M1, M2, N):
    """
    Calculate cross-covariance between data1 and data2
    If data1 == data2, calculates auto-covariance
    
    M1: number of channels of data1 (uint8, ≤ 255)
    M2: number of channels of data2 (uint8, ≤ 255)
    N: number of samples (uint16, ≤ 65535)
    
    example:
    Data1(N, M1) - (1000, 9)
    Data2(N, M2) - (1000, 10)
    
    """
    # Calculate mean for data1
    for j in range(M1):
        mean1[j] = 0.0
        for i in range(N):
            mean1[j] += data1[i, j]
        mean1[j] /= float(N)

    # Calculate mean for data2
    for j in range(M2):
        mean2[j] = 0.0
        for i in range(N):
            mean2[j] += data2[i, j]
        mean2[j] /= float(N)

    # Center the data
    # copy data to avoid modifying the original data
    data1_centered = data1.copy()
    data2_centered = data2.copy()
    for i in range(N):
        for j in range(M1):
            data1_centered[i, j] -= mean1[j]
        for j in range(M2):
            data2_centered[i, j] -= mean2[j]

    # Compute covariance
    for i in range(M1):
        for j in range(M2):
            cov[i, j] = 0.0
            for k in range(N):
                cov[i, j] += data1_centered[k, i] * data2_centered[k, j]
            cov[i, j] /= float(N - 1)

#----------------------------------------------------------------
# Allo implementation of covariance calculation
#----------------------------------------------------------------
def covariance(type, m1, m2, n):
    """
    Allo implementation of covariance calculation
    
    EEG signal is 2D array (N, M) where N is the number of samples and M is the number of channels
    EEG signal's amplitude(μV) is float64 or float32
    M1 and M2 are the number of channels, uint8 is enough (max 255 channels)
    N is the number of samples, uint16 is enough (max 65535 samples)
    """
    def kernel_covariance[T: (float64, float32), M1: int32, M2: int32, N: int32](
        data1: "T[N, M1]",    # First input matrix
        data2: "T[N, M2]",    # Second input matrix
        mean1: "T[M1]",       # Mean of first matrix
        mean2: "T[M2]",       # Mean of second matrix
        cov: "T[M1, M2]"      # Output covariance matrix
    ):
        # Compute mean for data1
        for x in allo.grid(M1):
            total: T = 0.0
            for k in allo.grid(N):
                total += data1[k, x]
            mean1[x] = total / N

        # Compute mean for data2
        for x in allo.grid(M2):
            total: T = 0.0
            for k in allo.grid(N):
                total += data2[k, x]
            mean2[x] = total / N

        # Compute cross-covariance
        for i, j in allo.grid(M1, M2):
            covariance: T = 0.0
            for p in allo.grid(N):
                covariance += (data1[p, i] - mean1[i]) * (data2[p, j] - mean2[j])
            cov[i, j] = covariance / (N - 1)

    s = allo.customize(kernel_covariance, instantiate=[type, m1, m2, n])
    mod = s.build()
    return mod

#----------------------------------------------------------------
# Test covariance implementation against numpy's direct matrix operations
#----------------------------------------------------------------
def test_covariance():
    """
    Test covariance implementation against numpy's direct matrix operations
    """
    # Read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    
    # For CI test we use small problem size
    test_psize = "small"
    M1 = psize["covariance"][test_psize]["M1"]
    M2 = psize["covariance"][test_psize]["M2"]
    N = psize["covariance"][test_psize]["N"]
    
    # Generate test data
    data1 = np.random.randint(-10, 10, (N, M1)).astype(np.float64)  # EEG data
    data2 = np.random.randint(-10, 10, (N, M2)).astype(np.float64)  # Reference data
    
    # Initialize arrays
    mean1 = np.zeros(M1, dtype=np.float64)
    mean2 = np.zeros(M2, dtype=np.float64)
    cov = np.zeros((M1, M2), dtype=np.float64)
    mean1_ref = np.zeros(M1, dtype=np.float64)
    mean2_ref = np.zeros(M2, dtype=np.float64)
    cov_ref = np.zeros((M1, M2), dtype=np.float64)
    
    # Calculate reference result using numpy implementation
    covariance_np(data1.copy(), data2.copy(), mean1_ref, mean2_ref, cov_ref, M1, M2, N)
    
    # Calculate result using Allo implementation
    mod = covariance(float64, M1, M2, N)
    mod(data1, data2, mean1, mean2, cov)
    
    # Verify results
    np.testing.assert_allclose(mean1, mean1_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(mean2, mean2_ref, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(cov, cov_ref, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__]) 
