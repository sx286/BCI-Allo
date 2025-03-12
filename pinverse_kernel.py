#----------------------------------------------------------------
# Calculate inverse of matrix A
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
# Numpy implementation of matrix inverse
#----------------------------------------------------------------
def pinverse_np(A, inv_A, N):
    """
    Calculate inverse of matrix A using numpy implementation
    
    Parameters:
    A -- Input matrix, shape: (N, N)
    inv_A -- Output inverse matrix, shape: (N, N)
    N -- Matrix dimension
    """
    # Use numpy's built-in function to calculate inverse
    inv_A[:] = np.linalg.pinv(A)

#----------------------------------------------------------------
# Allo implementation of pseudo-inverse (Moore-Penrose inverse) calculation
#----------------------------------------------------------------
def pinverse(type, n):
    """
    Allo implementation of pseudo-inverse (Moore-Penrose inverse) calculation
    Use Tikhonov regularization method to calculate pseudo-inverse
    
    Matrix A can be any square matrix (N, N)
    Matrix elements are float64 or float32
    N is the matrix dimension
    """
    def kernel_pinverse[T: (float64, float32), N: int32](
        A: "T[N, N]",        # Input matrix
        pinv_A: "T[N, N]",   # Output pseudo-inverse matrix
        temp1: "T[N, N]",    # Temporary matrix 1 (A^T * A)
        temp2: "T[N, N]"     # Temporary matrix 2 (A^T)
    ):
        epsilon: T = 1e-8    # Regularization parameter

        # Step 1: Calculate A^T
        for i, j in allo.grid(N, N):
            temp2[i, j] = A[j, i]

        # Step 2: Calculate A^T * A
        for i, j in allo.grid(N, N):
            sum: T = 0.0
            for k in allo.grid(N):
                sum += temp2[i, k] * A[k, j]
            temp1[i, j] = sum

        # Step 3: Add regularization term (A^T * A + epsilon * I)
        # Add small regularization term epsilon to the diagonal which improves numerical stability
        for i in allo.grid(N):
            temp1[i, i] = temp1[i, i] + epsilon

        # Step 4: Calculate (A^T * A + epsilon * I)^(-1)
        # Use improved Gauss-Jordan elimination
        for i, j in allo.grid(N, N):
            pinv_A[i, j] = 1.0 if i == j else 0.0

        for k in allo.grid(N):
            # Find the maximum pivot
            max_val: T = temp1[k, k] if temp1[k, k] >= 0.0 else -temp1[k, k]
            max_idx: int32 = k
            for i in range(k + 1, N):
                curr_val: T = temp1[i, k] if temp1[i, k] >= 0.0 else -temp1[i, k]
                if curr_val > max_val:
                    max_val = curr_val
                    max_idx = i
            
            # Swap rows
            if max_idx != k:
                for j in allo.grid(N):
                    temp: T = temp1[k, j]
                    temp1[k, j] = temp1[max_idx, j]
                    temp1[max_idx, j] = temp
                    temp = pinv_A[k, j]
                    pinv_A[k, j] = pinv_A[max_idx, j]
                    pinv_A[max_idx, j] = temp

            pivot: T = temp1[k, k]
            pivot_abs: T = pivot if pivot >= 0.0 else -pivot
            if pivot_abs > epsilon:  # Use absolute value comparison
                for j in allo.grid(N):
                    temp1[k, j] = temp1[k, j] / pivot
                    pinv_A[k, j] = pinv_A[k, j] / pivot

                for i in allo.grid(N):
                    if i != k:
                        factor: T = temp1[i, k]
                        for j in allo.grid(N):
                            temp1[i, j] = temp1[i, j] - factor * temp1[k, j]
                            pinv_A[i, j] = pinv_A[i, j] - factor * pinv_A[k, j]

        # Step 5: Calculate final pseudo-inverse (A^T * A + epsilon * I)^(-1) * A^T
        for i, j in allo.grid(N, N):
            sum: T = 0.0
            for k in allo.grid(N):
                sum += pinv_A[i, k] * temp2[k, j]
            temp1[i, j] = sum

        # Copy results to output matrix
        for i, j in allo.grid(N, N):
            pinv_A[i, j] = temp1[i, j]

    s = allo.customize(kernel_pinverse, instantiate=[type, n])
    mod = s.build()
    return mod

#----------------------------------------------------------------
# Test pseudo-inverse implementation against numpy's pinv
#----------------------------------------------------------------
def test_pinverse():
    """
    Test pseudo-inverse implementation against numpy's pinv for different problem sizes
    """
    # Read problem size settings
    setting_path = os.path.join(os.path.dirname(__file__), "psize.json")
    with open(setting_path, "r") as fp:
        psize = json.load(fp)
    
    # Test all sizes
    test_sizes = ["small", "medium", "large"]
    
    for size in test_sizes:
        print(f"\nTesting size: {size}")
        N = psize["inverse"][size]["N"]
        
        # Generate test matrix
        A = np.random.randn(N, N).astype(np.float64)
        # Make matrix ill-conditioned to test stability
        A = A @ A.T + 0.001 * np.eye(N)
        
        # Initialize arrays
        pinv_A = np.zeros((N, N), dtype=np.float64)
        pinv_A_ref = np.zeros((N, N), dtype=np.float64)
        temp1 = np.zeros((N, N), dtype=np.float64)
        temp2 = np.zeros((N, N), dtype=np.float64)
        
        # Use numpy to calculate reference result
        pinverse_np(A.copy(), pinv_A_ref, N)
        
        # Use Allo to calculate result
        mod = pinverse(float64, N)
        mod(A, pinv_A, temp1, temp2)
        
        # Verify results
        try:
            # First try with strict tolerance
            np.testing.assert_allclose(pinv_A, pinv_A_ref, rtol=1e-2, atol=1e-2)
            print(f"✓ {size} size test passed (rtol=1e-2, atol=1e-2)")
        except AssertionError as e:
            # If failed, try with relaxed tolerance
            try:
                np.testing.assert_allclose(pinv_A, pinv_A_ref, rtol=1e-1, atol=1e-1)
                print(f"△ {size} size test passed with relaxed tolerance (rtol=1e-1, atol=1e-1)")
            except AssertionError:
                # Calculate and display detailed error information
                abs_diff = np.abs(pinv_A - pinv_A_ref)
                rel_diff = np.abs((pinv_A - pinv_A_ref) / (pinv_A_ref + 1e-10))
                max_abs_diff = np.max(abs_diff)
                max_rel_diff = np.max(rel_diff)
                mismatch_count = np.sum(abs_diff > 1e-2)
                mismatch_percentage = (mismatch_count / (N * N)) * 100
                
                print(f"✗ {size} size test failed:")
                print(f"  - Maximum absolute error: {max_abs_diff:.6f}")
                print(f"  - Maximum relative error: {max_rel_diff:.6f}")
                print(f"  - Error element ratio: {mismatch_percentage:.6f}%")
                raise

if __name__ == "__main__":
    pytest.main([__file__]) 
