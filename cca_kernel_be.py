#---------------------------------------------------------------------------------
# CCA algorithm implemented in Allo (backend)
#---------------------------------------------------------------------------------
# CCA_kernel_be.py
# r is a 1D array of size 2

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
    def kernel_covariance[T: (float64, float32), N: int32, M1: int32, M2: int32](
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
    def kernel_pinverse[T: (float64, float32), M: int32](
        A: "T[M, M]",        # Input matrix
        pinv_A: "T[M, M]",   # Output pseudo-inverse matrix
        temp1: "T[M, M]",    # Temporary matrix 1 (A^T * A)
        temp2: "T[M, M]"     # Temporary matrix 2 (A^T)
    ):
        epsilon: T = 1e-8    # Regularization parameter

        # Step 1: Calculate A^T
        for i, j in allo.grid(M, M):
            temp2[i, j] = A[j, i]

        # Step 2: Calculate A^T * A
        for i, j in allo.grid(M, M):
            sum: T = 0.0
            for k in allo.grid(M):
                sum += temp2[i, k] * A[k, j]
            temp1[i, j] = sum

        # Step 3: Add regularization term (A^T * A + epsilon * I)
        for i in allo.grid(M):
            temp1[i, i] = temp1[i, i] + epsilon

        # Step 4: Calculate (A^T * A + epsilon * I)^(-1)
        for i, j in allo.grid(M, M):
            pinv_A[i, j] = 1.0 if i == j else 0.0

        for k in allo.grid(M):
            # Find the maximum pivot
            max_val: T = temp1[k, k] if temp1[k, k] >= 0.0 else -temp1[k, k]
            max_idx: int32 = k
            for i in range(k + 1, M):
                curr_val: T = temp1[i, k] if temp1[i, k] >= 0.0 else -temp1[i, k]
                if curr_val > max_val:
                    max_val = curr_val
                    max_idx = i
            
            # Swap rows
            if max_idx != k:
                for j in allo.grid(M):
                    temp: T = temp1[k, j]
                    temp1[k, j] = temp1[max_idx, j]
                    temp1[max_idx, j] = temp
                    temp = pinv_A[k, j]
                    pinv_A[k, j] = pinv_A[max_idx, j]
                    pinv_A[max_idx, j] = temp

            pivot: T = temp1[k, k]
            pivot_abs: T = pivot if pivot >= 0.0 else -pivot
            if pivot_abs > epsilon:
                for j in allo.grid(M):
                    temp1[k, j] = temp1[k, j] / pivot
                    pinv_A[k, j] = pinv_A[k, j] / pivot

                for i in allo.grid(M):
                    if i != k:
                        factor: T = temp1[i, k]
                        for j in allo.grid(M):
                            temp1[i, j] = temp1[i, j] - factor * temp1[k, j]
                            pinv_A[i, j] = pinv_A[i, j] - factor * pinv_A[k, j]

        # Step 5: Calculate final pseudo-inverse
        for i, j in allo.grid(M, M):
            sum: T = 0.0
            for k in allo.grid(M):
                sum += pinv_A[i, k] * temp2[k, j]
            temp1[i, j] = sum

        # Copy results to output matrix
        for i, j in allo.grid(M, M):
            pinv_A[i, j] = temp1[i, j]

    #---------------------------------------------------------------------------------
    # Eigenvalue kernel - 使用优化的幂法计算最大特征值
    #---------------------------------------------------------------------------------
    def kernel_eigenvalue[T: (float64, float32), M: int32](
        A: "T[M, M]",          # Input matrix
        eigenvals: "T[M]",     # Output array (only first element used)
        Q: "T[M, M]",         # Temporary workspace for eigenvector
        R: "T[M, M]"          # Temporary workspace
    ):
        max_iter: int32 = 50   # Increase iteration count for better precision
        epsilon: T = 1e-10     # Convergence threshold
        
        # Initialize to all ones vector - works well for symmetric positive definite matrices in CCA
        for i in allo.grid(M):
            Q[i, 0] = 1.0
        
        # Initial normalization
        norm_init: T = 0.0
        for i in allo.grid(M):
            norm_init += Q[i, 0] * Q[i, 0]
        norm_init = (norm_init ** 0.5) + epsilon
        
        for i in allo.grid(M):
            Q[i, 0] /= norm_init
        
        # Power method iteration
        for iter in allo.grid(max_iter):
            # Backup current vector
            for i in allo.grid(M):
                R[i, 1] = Q[i, 0]
            
            # Matrix-vector multiplication, using double precision accumulation
            for i in allo.grid(M):
                sum: T = 0.0
                for j in allo.grid(M):
                    sum += A[i, j] * Q[j, 0]
                R[i, 0] = sum
            
            # Calculate vector norm
            norm: T = 0.0
            for i in allo.grid(M):
                norm += R[i, 0] * R[i, 0]
            norm = (norm ** 0.5) + epsilon
            
            # Keep vector direction consistency
            dot_product: T = 0.0
            for i in allo.grid(M):
                dot_product += R[i, 0] * R[i, 1]
            
            sign: T = 1.0
            if dot_product < 0.0:
                sign = -1.0
            
            # Update vector
            for i in allo.grid(M):
                Q[i, 0] = sign * R[i, 0] / norm
        
        # Calculate Rayleigh quotient, get most accurate eigenvalue estimate
        numerator: T = 0.0
        denominator: T = 0.0
        
        for i in allo.grid(M):
            temp: T = 0.0
            for j in allo.grid(M):
                temp += A[i, j] * Q[j, 0]
            numerator += Q[i, 0] * temp
            denominator += Q[i, 0] * Q[i, 0]
        
        # Save maximum eigenvalue
        eigenvals[0] = numerator / (denominator + epsilon)
        
        # Zero other values
        for i in allo.grid(M-1):
            eigenvals[i+1] = 0.0

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
    # Square root kernel
    #---------------------------------------------------------------------------------
    def kernel_sqrt[T: (float64, float32), M: int32](
        eigenvals: "T[M]",     # Input eigenvalues array
        r: "T[2]"              # Output result (only first element used)
    ):
        # Get eigenvalues
        val: T = eigenvals[0]
        
        # Get sign
        sign: T = 1.0
        if val < 0.0:
            sign = -1.0
        
        # Get absolute value
        abs_val: T = val
        if val < 0.0:
            abs_val = -val
        
        # Calculate square root and keep sign
        r[0] = sign * (abs_val ** 0.5)
        r[1] = 0.0  # Initialize second element to 0

    #---------------------------------------------------------------------------------
    # Sort and signed square root kernel
    #---------------------------------------------------------------------------------
    # def kernel_sort_and_sqrt[T: (float64, float32), M: int32](
    #     A: "T[M]",           # Input array
    #     r: "T[2]"           # Output correlation coefficient, size changed to 2
    # ):
    #     # Step 1: Find maximum value
    #     max_val: T = -1.0
    #     for i in allo.grid(M):
    #         if A[i] > max_val:
    #             max_val = A[i]
        
    #     # Step 2: Calculate signed square root
    #     sign: T = 1
    #     if max_val < 0:
    #         sign = -1
            
    #     abs_val: T = max_val if max_val >= 0 else -max_val
        
    #     # Store result in first element, leave second element as 0
    #     r[0] = sign * abs_val ** 0.5
    #     r[1] = 0.0  # Initialize second element to 0

    #---------------------------------------------------------------------------------
    # 2. Main kernel
    #---------------------------------------------------------------------------------
    def kernel_cca[T: (float64, float32), N: int32, M1: int32, M2: int32](
        X: "T[N, M1]",      # First input matrix
        Y: "T[N, M2]",      # Second input matrix
        r: "T[2]"           # Correlation coefficients, size changed to 2
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
        Q: "T[M1, M1]"              # [M1, M1]
        R: "T[M1, M1]"              # [M1, M1]
        
        kernel_transpose[T, N, M1,"trans_x"](X, X_T)                         # [N,M1] -> [M1,N]
        kernel_transpose[T, N, M2,"trans_y"](Y, Y_T)                         # [N,M2] -> [M2,N]
        kernel_covariance[T, N, M1, M1, "cov_xx"](X, X, X_mean, X_mean, Cxx)  # [N,M1] @ [N,M1] -> [M1,M1]
        kernel_covariance[T, N, M2, M2, "cov_yy"](Y, Y, Y_mean, Y_mean, Cyy)  # [N,M2] @ [N,M2] -> [M2,M2]
        kernel_covariance[T, N, M1, M2, "cov_xy"](X, Y, X_mean, Y_mean, Cxy)  # [N,M1] @ [N,M2] -> [M1,M2]    
        kernel_transpose[T, M1, M2,"trans_cxy"](Cxy, Cyx)                    # [M1,M2] -> [M2,M1]
        kernel_pinverse[T, M1,"pinv_xx"](Cxx, Cxx_inv, temp1_M1, temp2_M1)   # [M1,M1] @ [M1,M1] -> [M1,M1]
        kernel_pinverse[T, M2,"pinv_yy"](Cyy, Cyy_inv, temp1_M2, temp2_M2)   # [M2,M2] @ [M2,M2] -> [M2,M2]
        kernel_gemm[T, M1, M1, M2,"gemm_xx_xy"](Cxx_inv, Cxy, temp3_M1)      # [M1,M1] @ [M1,M2] -> [M1,M2]
        kernel_gemm[T, M1, M2, M2,"gemm_xy_yy"](temp3_M1, Cyy_inv, temp4_M1) # [M1,M2] @ [M2,M2] -> [M1,M2]  
        kernel_gemm[T, M1, M2, M1,"gemm_xy_yx"](temp4_M1, Cyx, M)            # [M1,M2] @ [M2,M1] -> [M1,M1]
        kernel_eigenvalue[T, M1,"eigen_m"](M, eigenvals, Q, R)               # [M1,M1] @ [M1,M1] -> [M1]
        kernel_sqrt[T, M1,"sqrt"](eigenvals, r)                              # [M1] -> [2]

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
    s12 = allo.customize(kernel_eigenvalue, instantiate=[concrete_type, M1])        # [M1,M1] @ [M1,M1] -> [M1]
    # Square root kernel
    s13 = allo.customize(kernel_sqrt, instantiate=[concrete_type, M1])              # [M1] -> [2]

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
    sch.compose(s13, id="sqrt")

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
    r -- Maximum canonical correlation coefficient as a 1D array of size 2
    """
    # Center the data
    X = X - np.mean(X, axis=0)
    Y = Y - np.mean(Y, axis=0)
    
    # Initialize CCA with n_components=1
    cca = SklearnCCA(n_components=1)
    
    try:
        # Fit and transform the data
        cca.fit(X, Y)
        X_c, Y_c = cca.transform(X, Y)
        
        # Calculate correlation
        r = np.corrcoef(X_c.T, Y_c.T)[0, 1]
        # Return as array of size 2, with second element as 0
        return np.array([abs(r), 0.0], dtype=np.float32)
    except Exception as e:
        print(f"Error in sklearn CCA calculation: {e}")
        return np.array([0.0, 0.0], dtype=np.float32)

#================================================================================
# Test CCA algorithm using Vitis HLS
#================================================================================
# Define matrix dimensions
N = 100   # Number of samples
M1 = 9     # EEG channels
M2 = 10    # Reference signals

# CCA algorithm schedule
concrete_type = float32

sch = cca_algorithm(concrete_type, N, M1, M2)

# Generate Vitis HLS code and synthesize
# print("Start generating Vitis HLS code and synthesizing...")
# mod = sch.build() # using llvm
# mod = sch.build(target="vitis_hls",mode="csim",project="cca_0056.prj")
# mod = sch.build(target="vitis_hls",mode="hw_emu",project="cca_0056.prj")
mod = sch.build(target="vitis_hls",mode="hw",project="cca_0121.prj")

# # Generate random test data
# print("Generating test data...")
# np.random.seed(42)  # Set random seed to ensure reproducibility
X = np.random.rand(N, M1).astype(np.float32)  # EEG signals
Y = np.random.rand(N, M2).astype(np.float32)  # Reference signals
r_allo = np.zeros((2,), dtype=np.float32)  # Output correlation coefficient

# Calculate reference result (using sklearn implementation)
# print("Calculating reference result...")
r_ref = np.zeros((2,), dtype=np.float32)
r_ref = CCA_sklearn(X.copy(), Y.copy())

# Use generated hardware design for calculation
# print("Using hardware design for calculation...")
mod(X, Y, r_allo)
np.testing.assert_allclose(r_allo, r_ref, rtol=1e-2, atol=1e-3)

print("r_allo:", r_allo)
print("r_ref:", r_ref)


# def test_cca_vhls():
#     # Define matrix dimensions
#     N = 10   # Number of samples
#     M1 = 9     # EEG channels
#     M2 = 10    # Reference signals

#     # CCA algorithm schedule
#     concrete_type = float32

#     sch = cca_algorithm(concrete_type, N, M1, M2)

#     # Generate Vitis HLS code and synthesize
#     # print("Start generating Vitis HLS code and synthesizing...")
#     # mod = sch.build() # using llvm
#     # mod = sch.build(target="vitis_hls",mode="csim",project="cca.prj")
#     mod = sch.build(target="vitis_hls",mode="hw_emu",project="cca_1459.prj")

#     # # Generate random test data
#     # print("Generating test data...")
#     np.random.seed(42)  # Set random seed to ensure reproducibility
#     X = np.random.rand(N, M1).astype(np.float32)  # EEG signals
#     Y = np.random.rand(N, M2).astype(np.float32)  # Reference signals
#     r_allo = np.zeros((2,), dtype=np.float32)  # Output correlation coefficient

#     # Calculate reference result (using sklearn implementation)
#     # print("Calculating reference result...")
#     r_ref = np.zeros((2,), dtype=np.float32)
#     r_ref = CCA_sklearn(X.copy(), Y.copy())

#     # Use generated hardware design for calculation
#     # print("Using hardware design for calculation...")
#     mod(X, Y, r_allo)
#     np.testing.assert_allclose(r_allo, r_ref, rtol=1e-2, atol=1e-3)
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

# import pytest
# if __name__ == "__main__":
#     pytest.main([__file__])
#     # mod = test_cca_vhls()

#================================================================================
# Test the output type of the hardware design
#================================================================================
# Define matrix dimensions
# N = 10   # Number of samples
# M1 = 9     # EEG channels
# M2 = 10    # Reference signals

# # CCA algorithm schedule
# concrete_type = float32

# sch = cca_algorithm(concrete_type, N, M1, M2)
# mod= sch.build()
# # mod = sch.build(target="vitis_hls",mode="csim",project="cca_411_1133.prj")
# # mod = sch.build(target="vitis_hls",mode="hw_emu",project="cca_411_1136.prj")

# # # Generate random test data
# # print("Generating test data...")
# np.random.seed(42)  # Set random seed to ensure reproducibility
# X = np.random.rand(N, M1).astype(np.float32)  # EEG signals
# Y = np.random.rand(N, M2).astype(np.float32)  # Reference signals
# r_allo = np.zeros((2,), dtype=np.float32)  # Output correlation coefficient

# # Calculate reference result (using sklearn implementation)
# # print("Calculating reference result...")
# r_ref = np.zeros((2,), dtype=np.float32)
# r_ref = CCA_sklearn(X.copy(), Y.copy())

# # Use generated hardware design for calculation
# # print("Using hardware design for calculation...")

# print("\nBefore kernel call:")
# print("X dtype:", X.dtype, "nbytes:", X.itemsize)
# print("Y dtype:", Y.dtype, "nbytes:", Y.itemsize)
# print("r_allo dtype:", r_allo.dtype, "nbytes:", r_allo.itemsize)
# print("r_ref dtype:", r_ref.dtype, "nbytes:", r_ref.itemsize)

# X = np.ascontiguousarray(X, dtype=np.float32)
# Y = np.ascontiguousarray(Y, dtype=np.float32)
# r_allo = np.ascontiguousarray(r_allo, dtype=np.float32)

# mod(X, Y, r_allo)

# print("After kernel call:")
# print("\nAllo result:")
# print("r_allo value:", r_allo)
# print("r_allo type:", type(r_allo))
# print("r_allo dtype:", r_allo.dtype)
# print("r_allo shape:", r_allo.shape)
# print("r_allo nbytes:", r_allo.nbytes)

# print("\nReference result:")
# print("r_ref value:", r_ref)
# print("r_ref type:", type(r_ref))
# print("r_ref dtype:", r_ref.dtype)
# print("r_ref shape:", r_ref.shape)
# print("r_ref nbytes:", r_ref.nbytes)
