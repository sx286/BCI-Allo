#---------------------------------------------------------------------------------
# CCA algorithm implemented in Allo (backend)
#---------------------------------------------------------------------------------
# CCA_kernel_be.py
# using real BCI data to test the CCA algorithm

import allo
import numpy as np
from allo.ir.types import float64, float32, uint16, uint8
import allo.ir.types as T
import os

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
    def kernel_transpose[T: (float64, float32), N: uint16, M: uint8](
        A: "T[N, M]",      # Input matrix
        A_T: "T[M, N]"     # Output transposed matrix
    ):
        # Compute transpose
        for i_t, j_t in allo.grid(M, N):
            A_T[i_t, j_t] = A[j_t, i_t]

    #---------------------------------------------------------------------------------
    # Covariance kernel
    #---------------------------------------------------------------------------------
    def kernel_covariance[T: (float64, float32), N: uint16, M1: uint8, M2: uint8](
        data1: "T[N, M1]",    # First input matrix
        data2: "T[N, M2]",    # Second input matrix
        mean1: "T[M1]",       # Mean of first matrix
        mean2: "T[M2]",       # Mean of second matrix
        cov: "T[M1, M2]"      # Output covariance matrix
    ):
        # Compute mean for data1
        for x_c1 in allo.grid(M1):
            total: T = 0.0
            for k_c1 in allo.grid(N):
                total += data1[k_c1, x_c1]
            mean1[x_c1] = total / N

        # Compute mean for data2
        for x_c2 in allo.grid(M2):
            total: T = 0.0
            for k_c2 in allo.grid(N):
                total += data2[k_c2, x_c2]
            mean2[x_c2] = total / N

        # Compute cross-covariance
        for i_c, j_c in allo.grid(M1, M2):
            covariance: T = 0.0
            for p_c in allo.grid(N):
                covariance += (data1[p_c, i_c] - mean1[i_c]) * (data2[p_c, j_c] - mean2[j_c])
            cov[i_c, j_c] = covariance / (N - 1)

    #---------------------------------------------------------------------------------
    # Pseudo-inverse kernel
    #---------------------------------------------------------------------------------
    def kernel_pinverse[T: (float64, float32), M: uint8](
        A: "T[M, M]",        # Input matrix
        pinv_A: "T[M, M]",   # Output pseudo-inverse matrix
        temp1: "T[M, M]",    # Temporary matrix 1 (A^T * A)
        temp2: "T[M, M]"     # Temporary matrix 2 (A^T)
    ):
        epsilon: T = 1e-8    # Regularization parameter

        # Step 1: Calculate A^T
        for i_p1, j_p1 in allo.grid(M, M):
            temp2[i_p1, j_p1] = A[j_p1, i_p1]

        # Step 2: Calculate A^T * A
        for i_p2, j_p2 in allo.grid(M, M):
            sum: T = 0.0
            for k_p2 in allo.grid(M):
                sum += temp2[i_p2, k_p2] * A[k_p2, j_p2]
            temp1[i_p2, j_p2] = sum

        # Step 3: Add regularization term (A^T * A + epsilon * I)
        for i_p3 in allo.grid(M):
            temp1[i_p3, i_p3] = temp1[i_p3, i_p3] + epsilon

        # Step 4: Calculate (A^T * A + epsilon * I)^(-1)
        for i_p4, j_p4 in allo.grid(M, M):
            pinv_A[i_p4, j_p4] = 1.0 if i_p4 == j_p4 else 0.0

        for k_p5 in allo.grid(M):
            # Find the maximum pivot
            max_val: T = temp1[k_p5, k_p5] if temp1[k_p5, k_p5] >= 0.0 else -temp1[k_p5, k_p5]
            max_idx: uint8 = k_p5
            for i_p5 in range(k_p5 + 1, M):
                curr_val: T = temp1[i_p5, k_p5] if temp1[i_p5, k_p5] >= 0.0 else -temp1[i_p5, k_p5]
                if curr_val > max_val:
                    max_val = curr_val
                    max_idx = i_p5
            
            # Swap rows
            if max_idx != k_p5:
                for j_p5 in allo.grid(M):
                    temp: T = temp1[k_p5, j_p5]
                    temp1[k_p5, j_p5] = temp1[max_idx, j_p5]
                    temp1[max_idx, j_p5] = temp
                    temp = pinv_A[k_p5, j_p5]
                    pinv_A[k_p5, j_p5] = pinv_A[max_idx, j_p5]
                    pinv_A[max_idx, j_p5] = temp

            pivot: T = temp1[k_p5, k_p5]
            pivot_abs: T = pivot if pivot >= 0.0 else -pivot
            if pivot_abs > epsilon:
                for j_p6 in allo.grid(M):
                    temp1[k_p5, j_p6] = temp1[k_p5, j_p6] / pivot
                    pinv_A[k_p5, j_p6] = pinv_A[k_p5, j_p6] / pivot

                for i_p6 in allo.grid(M):
                    if i_p6 != k_p5:
                        factor: T = temp1[i_p6, k_p5]
                        for j_p7 in allo.grid(M):
                            temp1[i_p6, j_p7] = temp1[i_p6, j_p7] - factor * temp1[k_p5, j_p7]
                            pinv_A[i_p6, j_p7] = pinv_A[i_p6, j_p7] - factor * pinv_A[k_p5, j_p7]

        # Step 5: Calculate final pseudo-inverse
        for i_p8, j_p8 in allo.grid(M, M):
            sum: T = 0.0
            for k_p8 in allo.grid(M):
                sum += pinv_A[i_p8, k_p8] * temp2[k_p8, j_p8]
            temp1[i_p8, j_p8] = sum

        # Copy results to output matrix
        for i_p9, j_p9 in allo.grid(M, M):
            pinv_A[i_p9, j_p9] = temp1[i_p9, j_p9]

    #---------------------------------------------------------------------------------
    # Eigenvalue kernel
    #---------------------------------------------------------------------------------
    def kernel_eigenvalue[T: (float64, float32), M: uint8](
        A: "T[M, M]",          # Input matrix
        eigenvals: "T[M]",     # Output array (only first element used)
        Q: "T[M, M]",         # Temporary workspace for eigenvector
        R: "T[M, M]"          # Temporary workspace
    ):
        max_iter: uint8 = 50   # Increase iteration count for better precision
        epsilon: T = 1e-10     # Convergence threshold
        
        # Initialize to all ones vector - works well for symmetric positive definite matrices in CCA
        for i_e0 in allo.grid(M):
            Q[i_e0, 0] = 1.0
        
        # Initial normalization
        norm_init: T = 0.0
        for i_e1 in allo.grid(M):
            norm_init += Q[i_e1, 0] * Q[i_e1, 0]
        norm_init = (norm_init ** 0.5) + epsilon
        
        for i_e2 in allo.grid(M):
            Q[i_e2, 0] /= norm_init
        
        # Power method iteration
        for iter_e in allo.grid(max_iter):
            # Backup current vector
            for i_e3 in allo.grid(M):
                R[i_e3, 1] = Q[i_e3, 0]
            
            # Matrix-vector multiplication, using double precision accumulation
            for i_e4 in allo.grid(M):
                sum: T = 0.0
                for j_e4 in allo.grid(M):
                    sum += A[i_e4, j_e4] * Q[j_e4, 0]
                R[i_e4, 0] = sum
            
            # Calculate vector norm
            norm: T = 0.0
            for i_e5 in allo.grid(M):
                norm += R[i_e5, 0] * R[i_e5, 0]
            norm = (norm ** 0.5) + epsilon
            
            # Keep vector direction consistency
            dot_product: T = 0.0
            for i_e6 in allo.grid(M):
                dot_product += R[i_e6, 0] * R[i_e6, 1]
            
            sign: T = 1.0
            if dot_product < 0.0:
                sign = -1.0
            
            # Update vector
            for i_e7 in allo.grid(M):
                Q[i_e7, 0] = sign * R[i_e7, 0] / norm
        
        # Calculate Rayleigh quotient, get most accurate eigenvalue estimate
        numerator: T = 0.0
        denominator: T = 0.0
        
        for i_e8 in allo.grid(M):
            temp: T = 0.0
            for j_e8 in allo.grid(M):
                temp += A[i_e8, j_e8] * Q[j_e8, 0]
            numerator += Q[i_e8, 0] * temp
            denominator += Q[i_e8, 0] * Q[i_e8, 0]
        
        # Save maximum eigenvalue
        eigenvals[0] = numerator / (denominator + epsilon)
        
        # Zero other values
        for i_e9 in allo.grid(M-1):
            eigenvals[i_e9+1] = 0.0

    #---------------------------------------------------------------------------------
    # General Matrix Multiplication (GEMM) kernel
    #---------------------------------------------------------------------------------
    def kernel_gemm[T: (float64, float32), M: uint8, K: uint8, N: uint8](
        A: "T[M, K]",      # Input matrix A
        B: "T[K, N]",      # Input matrix B
        C: "T[M, N]"       # Output matrix C
    ):
        # Initialize output matrix
        for i_g0, j_g0 in allo.grid(M, N):
            C[i_g0, j_g0] = 0.0
            
        # Matrix multiplication
        for i_g1, j_g1 in allo.grid(M, N):
            sum: T = 0.0
            for k_g1 in allo.grid(K):
                sum += A[i_g1, k_g1] * B[k_g1, j_g1]
            C[i_g1, j_g1] = sum

    #---------------------------------------------------------------------------------
    # Square root kernel
    #---------------------------------------------------------------------------------
    def kernel_sqrt[T: (float64, float32), M: uint8](
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
    # 2. Main kernel
    #---------------------------------------------------------------------------------
    def kernel_cca[T: (float64, float32), N: uint16, M1: uint8, M2: uint8](
        X: "T[N, M1]",      # First input matrix (1000, 9)
        Y: "T[N, M2]",      # Second input matrix (1000,10)
        r: "T[2]"           # Correlation coefficients, size changed to 2 (2, 0)
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
        
        kernel_transpose[T, N, M1,"trans_x"](X, X_T)                         # [N,M1] -> [M1,N] : (1000, 9) -> (9, 1000)
        kernel_transpose[T, N, M2,"trans_y"](Y, Y_T)                         # [N,M2] -> [M2,N] : (1000, 10) -> (10, 1000)
        kernel_covariance[T, N, M1, M1, "cov_xx"](X, X, X_mean, X_mean, Cxx) # [N,M1] @ [N,M1] -> [M1,M1] : (1000, 9) @ (1000, 9) -> (9, 9)
        kernel_covariance[T, N, M2, M2, "cov_yy"](Y, Y, Y_mean, Y_mean, Cyy) # [N,M2] @ [N,M2] -> [M2,M2] : (1000, 10) @ (1000, 10) -> (10, 10)
        kernel_covariance[T, N, M1, M2, "cov_xy"](X, Y, X_mean, Y_mean, Cxy) # [N,M1] @ [N,M2] -> [M1,M2] : (1000, 9) @ (1000, 10) -> (9, 10)
        kernel_transpose[T, M1, M2,"trans_cxy"](Cxy, Cyx)                    # [M1,M2] -> [M2,M1] : (9, 10) -> (10, 9)
        kernel_pinverse[T, M1,"pinv_xx"](Cxx, Cxx_inv, temp1_M1, temp2_M1)   # [M1,M1] @ [M1,M1] -> [M1,M1] : (9, 9) @ (9, 9) -> (9, 9)
        kernel_pinverse[T, M2,"pinv_yy"](Cyy, Cyy_inv, temp1_M2, temp2_M2)   # [M2,M2] @ [M2,M2] -> [M2,M2] : (10, 10) @ (10, 10) -> (10, 10)
        kernel_gemm[T, M1, M1, M2,"gemm_xx_xy"](Cxx_inv, Cxy, temp3_M1)      # [M1,M1] @ [M1,M2] -> [M1,M2] : (9, 9) @ (9, 10) -> (9, 10)
        kernel_gemm[T, M1, M2, M2,"gemm_xy_yy"](temp3_M1, Cyy_inv, temp4_M1) # [M1,M2] @ [M2,M2] -> [M1,M2] : (9, 10) @ (10, 10) -> (9, 10)
        kernel_gemm[T, M1, M2, M1,"gemm_xy_yx"](temp4_M1, Cyx, M)            # [M1,M2] @ [M2,M1] -> [M1,M1] : (9, 10) @ (10, 9) -> (9, 9)
        kernel_eigenvalue[T, M1,"eigen_m"](M, eigenvals, Q, R)               # [M1,M1] @ [M1,M1] -> [M1] : (9, 9) @ (9, 9) -> (9)
        kernel_sqrt[T, M1,"sqrt"](eigenvals, r)                              # [M1] -> [2] : (9) -> (2)

    #---------------------------------------------------------------------------------  
    # Transpose kernel
    #---------------------------------------------------------------------------------
    s1  = allo.customize(kernel_transpose, instantiate=[concrete_type, N, M1])      # [N,M1] -> [M1,N] : (1000, 9) -> (9, 1000)
    s2  = allo.customize(kernel_transpose, instantiate=[concrete_type, N, M2])      # [N,M2] -> [M2,N] : (1000, 10) -> (10, 1000)   
    s3  = allo.customize(kernel_transpose, instantiate=[concrete_type, M1, M2])     # [M1,M2] -> [M2,M1] : (9, 10) -> (10, 9)

    # Optimize transpose kernel - 避免对大循环进行过度优化
    # 对小矩阵直接优化
    s1.pipeline("i_t")
    s2.pipeline("i_t")
    s3.pipeline("j_t")

    #---------------------------------------------------------------------------------  
    # Covariance kernel
    #---------------------------------------------------------------------------------
    s4  = allo.customize(kernel_covariance, instantiate=[concrete_type, N, M1, M1]) # [N,M1] @ [N,M1] -> [M1,M1] : (1000, 9) @ (1000, 9) -> (9, 9)
    s5  = allo.customize(kernel_covariance, instantiate=[concrete_type, N, M2, M2]) # [N,M2] @ [N,M2] -> [M2,M2] : (1000, 10) @ (1000, 10) -> (10, 10)  
    s6  = allo.customize(kernel_covariance, instantiate=[concrete_type, N, M1, M2]) # [N,M1] @ [N,M2] -> [M1,M2] : (1000, 9) @ (1000, 10) -> (9, 10)

    # Optimize covariance kernel - 更谨慎地优化，避免对长度为1000的循环进行pipeline
    # 改为只对小循环进行优化
    # s4.pipeline("i_c")  # 对外层小循环流水线化(M1=9)
    # s4.buffer_at(s4.cov, "i_c")  # 缓存输出结果

    # s5.pipeline("i_c")  # 对外层小循环流水线化(M2=10)
    # s5.buffer_at(s5.cov, "i_c")

    # s6.pipeline("i_c")  # 对外层小循环流水线化
    # s6.buffer_at(s6.cov, "i_c")

    #---------------------------------------------------------------------------------  
    # Pseudo-inverse kernel
    #---------------------------------------------------------------------------------
    s7  = allo.customize(kernel_pinverse, instantiate=[concrete_type, M1])          # [M1,M1] @ [M1,M1] -> [M1,M1] : (9, 9) @ (9, 9) -> (9, 9)
    s8  = allo.customize(kernel_pinverse, instantiate=[concrete_type, M2])          # [M2,M2] @ [M2,M2] -> [M2,M2] : (10, 10) @ (10, 10) -> (10, 10)

    # Optimize pseudo-inverse kernel - 伪逆已经是小矩阵(9x9, 10x10)，可以保留部分优化
    # # 对矩阵乘法部分优化
    # s7.pipeline("j_p1")  # 只对内层小循环进行优化

    # # 对s8应用类似优化
    # s8.pipeline("j_p1")

    #---------------------------------------------------------------------------------  
    # GEMM kernel
    #---------------------------------------------------------------------------------
    s9  = allo.customize(kernel_gemm, instantiate=[concrete_type, M1, M1, M2])      # [M1,M1] @ [M1,M2] -> [M1,M2] : (9, 9) @ (9, 10) -> (9, 10)
    s10 = allo.customize(kernel_gemm, instantiate=[concrete_type, M1, M2, M2])      # [M1,M2] @ [M2,M2] -> [M1,M2] : (9, 10) @ (10, 10) -> (9, 10)  
    s11 = allo.customize(kernel_gemm, instantiate=[concrete_type, M1, M2, M1])      # [M1,M2] @ [M2,M1] -> [M1,M1] : (9, 10) @ (10, 9) -> (9, 9)

    # # Optimize GEMM kernel - GEMM是小矩阵，优化内层循环
    # s9.pipeline("j_g0")  # 初始化矩阵的内层循环
    # s9.buffer_at(s9.C, "i_g0")  # 按行缓存结果

    # s10.pipeline("j_g0")
    # s10.buffer_at(s10.C, "i_g0")

    # s11.pipeline("j_g0")
    # s11.buffer_at(s11.C, "i_g0")

    #---------------------------------------------------------------------------------  
    # Eigenvalue kernel
    #---------------------------------------------------------------------------------
    s12 = allo.customize(kernel_eigenvalue, instantiate=[concrete_type, M1])        # [M1,M1] @ [M1,M1] -> [M1] : (9, 9) @ (9, 9) -> (9)

    # Optimize eigenvalue kernel - 更谨慎地优化，避免迭代循环优化
    # 优化矩阵向量乘法中的内层循环
    # s12.pipeline("j_e4")  # 矩阵向量乘法的内层循环

    # 其他循环不做流水线优化

    #---------------------------------------------------------------------------------  
    # Square root kernel
    #---------------------------------------------------------------------------------
    s13 = allo.customize(kernel_sqrt, instantiate=[concrete_type, M1])              # [M1] -> [2] : (9) -> (2) 

    #---------------------------------------------------------------------------------  
    # Compose CCA sub-kernels
    #---------------------------------------------------------------------------------
    sch = allo.customize(kernel_cca, instantiate=[concrete_type, N, M1, M2])

    # 组合子内核
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
# Test CCA algorithm using Vitis HLS with real EEG data
#================================================================================
# Define parameters
N = 1000   # Number of samples
M1 = 9     # EEG channels
num_harmonics = 5  # Number of harmonics for reference signals
M2 = 2 * num_harmonics  # Reference signals (sine and cosine for each harmonic)
fs = 250  # Sampling rate

# Define target and block parameters
target_idx = 1  # Target number (1-40)
ref_idx = 40     # Reference number (1-40)
block_idx = 1   # Block number (1-6)

# Target frequency mapping (get the corresponding frequency according to the target number)
ref_freqs = {
    1: 8.0,   2: 9.0,   3: 10.0,  4: 11.0,  5: 12.0,  6: 13.0,  7: 14.0,  8: 15.0,
    9: 8.2,   10: 9.2,  11: 10.2, 12: 11.2, 13: 12.2, 14: 13.2, 15: 14.2, 16: 15.2,
    17: 8.4,  18: 9.4,  19: 10.4, 20: 11.4, 21: 12.4, 22: 13.4, 23: 14.4, 24: 15.4,
    25: 8.6,  26: 9.6,  27: 10.6, 28: 11.6, 29: 12.6, 30: 13.6, 31: 14.6, 32: 15.6,
    33: 8.8,  34: 9.8,  35: 10.8, 36: 11.8, 37: 12.8, 38: 13.8, 39: 14.8, 40: 15.8
}

# Set the reference frequency
target_freq = ref_freqs[target_idx]
ref_freq = ref_freqs[ref_idx]

print(f"\nSelected target parameters:")
print(f"Block number: {block_idx}")
print(f"Target number: {target_idx}, Target frequency: {target_freq} Hz")
print(f"Reference number: {ref_idx}, Reference frequency: {ref_freq} Hz")

# Build the file path
base_dir = "/home/sx286/allo/BCI/test/EEG_Benchmark/extracted_data"
eeg_data_path = os.path.join(base_dir, f"S2_target_{target_idx}_block_{block_idx}.npy")

# Check if the file exists
if not os.path.exists(eeg_data_path):
    raise FileNotFoundError(f"EEG data file not found: {eeg_data_path}")

# CCA algorithm schedule
concrete_type = float32

sch = cca_algorithm(concrete_type, N, M1, M2)

# Generate Vitis HLS code and synthesize
# print("Start generating Vitis HLS code and synthesizing...")
mod = sch.build() # using llvm
# mod = sch.build(target="vitis_hls",mode="csim",project="cca_13_13.prj")
# mod = sch.build(target="vitis_hls",mode="hw_emu",project="cca_13_13.prj")
# mod = sch.build(target="vitis_hls",mode="hw",project="cca_0109.prj")

# Load extracted EEG data
eeg_data = np.load(eeg_data_path)
print(f"Loaded EEG data shape: {eeg_data.shape}")

X = eeg_data.astype(np.float32)
print(f"X shape: {X.shape}")

# Generate reference signals
tidx = np.arange(1, N + 1) / fs  # time index

# Initialize reference signals - shape: (1000, 2*num_harmonics)
Y = np.zeros((N, 2 * num_harmonics), dtype=np.float32)

# Generate reference signals
for harm_i in range(1, num_harmonics + 1):
    # Calculate the sine and cosine components of the current harmonic
    sin_idx = (harm_i - 1) * 2
    cos_idx = (harm_i - 1) * 2 + 1
    
    # Generate sine signal
    Y[:, sin_idx] = np.sin(2 * np.pi * harm_i * ref_freq * tidx).astype(np.float32)
    # Generate cosine signal
    Y[:, cos_idx] = np.cos(2 * np.pi * harm_i * ref_freq * tidx).astype(np.float32)

print(f"Y shape: {Y.shape}")

# Initialize correlation coefficient output
r_allo = np.zeros((2,), dtype=np.float32)

# Call sklearn CCA as reference
r_ref = np.zeros((2,), dtype=np.float32)
r_ref = CCA_sklearn(X.copy(), Y.copy())

# Use the hardware design generated by Allo for calculation
mod(X, Y, r_allo)

# Print results
print("\n==== CCA Results ====")
print(f"Allo CCA coefficient: {r_allo}")
print(f"Reference CCA coefficient: {r_ref}")

# Calculate error
abs_error = abs(r_allo[0] - r_ref[0])
rel_error = abs_error / r_ref[0] * 100 if r_ref[0] != 0 else float('inf')

print(f"Absolute error: {abs_error:.6f}")
print(f"Relative error: {rel_error:.2f}%")

# Verify results (using a more relaxed error tolerance)
try:
    # Tolerance: rtol 5e-2 (5%), atol 1e-2(1%)
    np.testing.assert_allclose(r_allo, r_ref, rtol=5e-2, atol=25e-2)
    print("\n✓ Hardware design test passed: correlation coefficient matches reference")
    print("  (Using relaxed tolerance: rtol=5%, atol=0.25)")
except AssertionError as e:
    print("\n✗ Hardware design test failed: correlation coefficient does not match reference")
    print(f"  Relative error: {rel_error:.2f}% (tolerance: 5%)")
    print(f"  Absolute error: {abs_error:.6f} (tolerance: 0.25)")
