#---------------------------------------------------------------------------------
# CCA algorithm implemented in Allo (backend)
#---------------------------------------------------------------------------------
# CCA_kernel_be.py
# Leverage the Allo DSL to generate Vivado/Vitis HLS C++ code for FPGA

import allo
import numpy as np
from allo.ir.types import int32, float64, float32
import allo.ir.types as T

#================================================================================
# CCA algorithm using Allo
#================================================================================
def cca_algorithm(concrete_type, N, M1, M2):
    """
    Create CCA algorithm with kernel composition
    
    Args:
        concrete_type: data type (float64/float32)
        N: number of samples
        M1: first dimension (number of channels)
        M2: second dimension (number of reference signals)
    """

    #================================================================================
    # 1.Sub-kernel definition (used in the top kernel)
    #================================================================================

    #---------------------------------------------------------------------------------
    # Transpose kernel
    #---------------------------------------------------------------------------------
    def kernel_transpose[T: (float64, float32), N: int32, M: int32](
        A: "T[N, M]",      # Input matrix
        A_T: "T[M, N]"     # Output transposed matrix
    ):
        # Compute transpose
        for i, j in allo.grid(M, N):
            A_T[i, j] = A[j, i]

    #---------------------------------------------------------------------------------
    # Covariance kernel
    #---------------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------------
    # Pseudo-inverse kernel
    #---------------------------------------------------------------------------------
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
        for i in allo.grid(N):
            temp1[i, i] = temp1[i, i] + epsilon

        # Step 4: Calculate (A^T * A + epsilon * I)^(-1)
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
            if pivot_abs > epsilon:
                for j in allo.grid(N):
                    temp1[k, j] = temp1[k, j] / pivot
                    pinv_A[k, j] = pinv_A[k, j] / pivot

                for i in allo.grid(N):
                    if i != k:
                        factor: T = temp1[i, k]
                        for j in allo.grid(N):
                            temp1[i, j] = temp1[i, j] - factor * temp1[k, j]
                            pinv_A[i, j] = pinv_A[i, j] - factor * pinv_A[k, j]

        # Step 5: Calculate final pseudo-inverse
        for i, j in allo.grid(N, N):
            sum: T = 0.0
            for k in allo.grid(N):
                sum += pinv_A[i, k] * temp2[k, j]
            temp1[i, j] = sum

        # Copy results to output matrix
        for i, j in allo.grid(N, N):
            pinv_A[i, j] = temp1[i, j]

    #---------------------------------------------------------------------------------
    # Eigenvalue kernel
    #---------------------------------------------------------------------------------
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

    #---------------------------------------------------------------------------------
    # General Matrix Multiplication (GEMM) kernel
    #---------------------------------------------------------------------------------
    def kernel_gemm[T: (float64, float32), M: int32, K: int32, N: int32](
        A: "T[M, K]",      # Input matrix A
        B: "T[K, N]",      # Input matrix B
        C: "T[M, N]"       # Output matrix C
    ):
        # Initialize output matrix
        for i, j in allo.grid(M, N):
            C[i, j] = 0.0
            
        # Matrix multiplication
        for i, j in allo.grid(M, N):
            sum: T = 0.0
            for k in allo.grid(K):
                sum += A[i, k] * B[k, j]
            C[i, j] = sum

    #---------------------------------------------------------------------------------
    # Sort kernel
    #---------------------------------------------------------------------------------
    def kernel_sort[T: (float64, float32), N: int32](
        A: "T[N]",           # Input array
        max_val: "T[1]"      # Output maximum value array (size 1)
    ):
        # Initialize to first element
        max_val[0] = A[0]
        
        # Use single grid to traverse and find maximum value
        for i in allo.grid(N):
            if A[i] > max_val[0]:
                max_val[0] = A[i]

    #----------------------------------------------------------------
    # Signed square root kernel
    #----------------------------------------------------------------
    def kernel_signed_sqrt[T: (float64, float32), N: int32](
        x: "T[1]",    # Input value
        y: "T[1]"     # Output value
    ):
        # Get sign of input
        sign: T = 1
        if x[0] < 0:
            sign = -1
            
        # Get absolute value
        abs_x: T = x[0] if x[0] >= 0 else -x[0]
        
        # Calculate square root using Newton's method
        y[0] = sign * abs_x ** 0.5  # Initial guess

    #---------------------------------------------------------------------------------
    # 2. Main kernel
    #---------------------------------------------------------------------------------
    def kernel_cca[T: (float64, float32), N: int32, M1: int32, M2: int32](
        X: "T[N, M1]",      # First input matrix
        Y: "T[N, M2]",      # Second input matrix
        r: "T[1]"           # Correlation coefficients
    ):
        X_T: "T[M1, N]"             # [M1, N]
        Y_T: "T[M2, N]"             # [M2, N]
        X_mean: "T[M1]"             # [M1]
        Y_mean: "T[M2]"             # [M2]
        Cxx: "T[M1, M1]"            # [M1, M1]
        Cyy: "T[M2, M2]"            # [M2, M2]
        Cxy: "T[M1, M2]"            # [M1, M2]
        Cyx: "T[M2, M1]"            # [M2, M1]
        Cxx_inv: "T[M1, M1]"        # [M1, M1]
        Cyy_inv: "T[M2, M2]"        # [M2, M2]
        temp1_M1: "T[M1, M1]"       # [M1, M1] for pinverse
        temp2_M1: "T[M1, M1]"       # [M1, M1] for pinverse
        temp3_M1: "T[M1, M2]"       # [M1, M2] for gemm
        temp4_M1: "T[M1, M2]"       # [M1, M2] for gemm
        temp1_M2: "T[M2, M2]"       # [M2, M2] for pinverse
        temp2_M2: "T[M2, M2]"       # [M2, M2] for pinverse
        M: "T[M1, M1]"              # [M1, M1]
        eigenvals: "T[M1]"          # [M1]
        max_eigenvals: "T[1]"       # [1]
        Q: "T[M1, M1]"              # [M1, M1]
        R: "T[M1, M1]"              # [M1, M1]
        
        kernel_transpose[T, N, M1,"trans_x"](X, X_T)                         # [N,M1] -> [M1,N]
        kernel_transpose[T, N, M2,"trans_y"](Y, Y_T)                         # [N,M2] -> [M2,N]
        kernel_covariance[T, M1, M1, N,"cov_xx"](X, X, X_mean, X_mean, Cxx)  # [N,M1] @ [N,M1] -> [M1,M1]
        kernel_covariance[T, M2, M2, N,"cov_yy"](Y, Y, Y_mean, Y_mean, Cyy)  # [N,M2] @ [N,M2] -> [M2,M2]
        kernel_covariance[T, M1, M2, N,"cov_xy"](X, Y, X_mean, Y_mean, Cxy)  # [N,M1] @ [N,M2] -> [M1,M2]    
        kernel_transpose[T, M1, M2,"trans_cxy"](Cxy, Cyx)                    # [M1,M2] -> [M2,M1]
        kernel_pinverse[T, M1,"pinv_xx"](Cxx, Cxx_inv, temp1_M1, temp2_M1)   # [M1,M1] @ [M1,M1] -> [M1,M1]
        kernel_pinverse[T, M2,"pinv_yy"](Cyy, Cyy_inv, temp1_M2, temp2_M2)   # [M2,M2] @ [M2,M2] -> [M2,M2]
        kernel_gemm[T, M1, M1, M2,"gemm_xx_xy"](Cxx_inv, Cxy, temp3_M1)      # [M1,M1] @ [M1,M2] -> [M1,M2]
        kernel_gemm[T, M1, M2, M2,"gemm_xy_yy"](temp3_M1, Cyy_inv, temp4_M1) # [M1,M2] @ [M2,M2] -> [M1,M2]  
        kernel_gemm[T, M1, M2, M1,"gemm_xy_yx"](temp4_M1, Cyx, M)            # [M1,M2] @ [M2,M1] -> [M1,M1]
        kernel_eigenvalue[T, M1,"eigen_m"](M, eigenvals, Q, R)               # [M1,M1] @ [M1,M1] -> [M1,M1]
        kernel_sort[T, M1,"sort_eigen"](eigenvals, max_eigenvals)            # [M1] -> [1]
        kernel_signed_sqrt[T, M1,"sqrt_eigen"](max_eigenvals, r)             # [1] -> [1]   

    # Create and optimize sub-kernels
    # Transpose kernel
    s1  = allo.customize(kernel_transpose, instantiate=[concrete_type, N, M1])      # [N,M1] -> [M1,N]
    s2  = allo.customize(kernel_transpose, instantiate=[concrete_type, N, M2])      # [N,M2] -> [M2,N]
    s3  = allo.customize(kernel_transpose, instantiate=[concrete_type, M1, M2])     # [M1,M2] -> [M2,M1]
    # Covariance kernel
    s4  = allo.customize(kernel_covariance, instantiate=[concrete_type, M1, M1, N]) # [N,M1] @ [N,M1] -> [M1,M1]
    s5  = allo.customize(kernel_covariance, instantiate=[concrete_type, M2, M2, N]) # [N,M2] @ [N,M2] -> [M2,M2]
    s6  = allo.customize(kernel_covariance, instantiate=[concrete_type, M1, M2, N]) # [N,M1] @ [N,M2] -> [M1,M2]
    # Pseudo-inverse kernel
    s7  = allo.customize(kernel_pinverse, instantiate=[concrete_type, M1])          # [M1,M1] @ [M1,M1] -> [M1,M1]
    s8  = allo.customize(kernel_pinverse, instantiate=[concrete_type, M2])          # [M2,M2] @ [M2,M2] -> [M2,M2]
    # GEMM kernel
    s9  = allo.customize(kernel_gemm, instantiate=[concrete_type, M1, M1, M2])      # [M1,M1] @ [M1,M2] -> [M1,M2]
    s10 = allo.customize(kernel_gemm, instantiate=[concrete_type, M1, M2, M2])      # [M1,M2] @ [M2,M2] -> [M1,M2]
    s11 = allo.customize(kernel_gemm, instantiate=[concrete_type, M1, M2, M1])      # [M1,M2] @ [M2,M1] -> [M1,M1]
    # Eigenvalue kernel
    s12 = allo.customize(kernel_eigenvalue, instantiate=[concrete_type, M1])        # [M1,M1] @ [M1,M1] -> [M1,M1]
    # Sort kernel
    s13 = allo.customize(kernel_sort, instantiate=[concrete_type, M1])              # [M1] -> [1]
    # Signed square root kernel
    s14 = allo.customize(kernel_signed_sqrt, instantiate=[concrete_type, 1])        # [1] -> [1]

    sch = allo.customize(kernel_cca, instantiate=[concrete_type, N, M1, M2])
    sch.compose(s1, id="trans_x")
    sch.compose(s2, id="trans_y")
    sch.compose(s3, id="trans_cxy")

    sch.compose(s4, id="cov_xx")
    sch.compose(s5, id="cov_yy")
    sch.compose(s6, id="cov_xy")

    sch.compose(s7, id="pinv_xx")
    sch.compose(s8, id="pinv_yy")

    sch.compose(s9, id="gemm_xx_xy")
    sch.compose(s10, id="gemm_xy_yy")
    sch.compose(s11, id="gemm_xy_yx")

    sch.compose(s12, id="eigen_m")
    sch.compose(s13, id="sort_eigen")
    sch.compose(s14, id="sqrt_eigen")

    return sch

#================================================================================
# CCA algorithm using sklearn
#================================================================================
from sklearn.cross_decomposition import CCA as SklearnCCA

def CCA_sklearn(X: np.ndarray, Y: np.ndarray):
    """
    Implementation of CCA algorithm using sklearn
    
    Parameters:
    X -- EEG signal data, shape: (num_samples, num_channels)
    Y -- Reference signal data, shape: (num_samples, num_harmonics)
    
    Returns:
    r -- Maximum canonical correlation coefficient
    """
    # Center the data
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    
    # Initialize CCA with n_components=1 as we only need the first correlation
    cca = SklearnCCA(n_components=1)
    
    try:
        # Fit and transform the data
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        
        # Calculate correlation
        r = np.corrcoef(X_c.T, Y_c.T)[0, 1]
        return abs(r)
    except Exception as e:
        print(f"Error in sklearn CCA calculation: {e}")
        return 0.0

#================================================================================
# Test CCA algorithm using Vitis HLS
#================================================================================
import pytest

def test_cca_vhls():
    # Define matrix dimensions
    N = 100   # Number of samples
    M1 = 9     # EEG channels
    M2 = 10    # Reference signals

    # CCA algorithm schedule
    concrete_type = float64
    sch = cca_algorithm(concrete_type, N, M1, M2)

    # Generate Vitis HLS code and synthesize
    # print("Start generating Vitis HLS code and synthesizing...")
    # mod = sch.build() # using llvm
    mod = sch.build(target="vitis_hls",mode="csim",project="cca.prj")
    # mod = sch.build(target="vitis_hls",mode="hw_emu",project="cca.prj")

    # # Generate random test data
    # print("Generating test data...")
    np.random.seed(42)  # Set random seed to ensure reproducibility
    X = np.random.rand(N, M1).astype(np.float64)  # EEG signals
    Y = np.random.rand(N, M2).astype(np.float64)  # Reference signals
    r_allo = np.zeros(1, dtype=np.float64)  # Output correlation coefficient

    # Calculate reference result (using sklearn implementation)
    # print("Calculating reference result...")
    r_ref = CCA_sklearn(X.copy(), Y.copy())

    # Use generated hardware design for calculation
    # print("Using hardware design for calculation...")
    mod(X, Y, r_allo)
    np.testing.assert_allclose(r_allo[0], r_ref, rtol=1e-2, atol=1e-3)
    # # Display results
    # print(f"\nInput data dimensions: X = {X.shape}, Y = {Y.shape}")
    # print(f"Hardware design calculation result: {r_allo[0]}")
    # print(f"NumPy reference result: {r_ref}")
    # print(f"Absolute error: {abs(r_allo[0] - r_ref)}")
   

    
    # Verify results
    # try:
    #     np.testing.assert_allclose(r_allo[0], r_ref, rtol=1e-2, atol=1e-3)
    #     print("\n✓ Hardware design test passed: correlation coefficient matches")
    # except AssertionError:
    #     print("\n✗ Hardware design test failed: correlation coefficient does not match")
    #     print(f"   Expected value: {r_ref}")
    #     print(f"   Actual value: {r_allo[0]}")
    #     print(f"   Error: {abs(r_allo[0] - r_ref)}")
    #     raise
    

if __name__ == "__main__":
    pytest.main([__file__])
    # mod = test_cca_vhls()
