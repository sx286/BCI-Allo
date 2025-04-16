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
def cca_algorithm(concrete_type, N, M1, M2, F):
    """
    Create SSVEP-CCA algorithm with kernel composition
    
    Args:
        concrete_type: data type (float64/float32)
        N: number of samples
        M1: number of EEG channels
        M2: number of reference signals: number of harmonics *2
        F: number of frequencies
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
    def kernel_mean[T: (float64, float32), N: uint16, M: uint8](
        data: "T[N, M]",    
        mean: "T[M]"  
    ):
        # Compute mean for data1 
        for x_c in allo.grid(M): # outer loop: M
            total: T = 0.0
            for k_c in allo.grid(N): # inner loop: N
                total += data[k_c, x_c]
            mean[x_c] = total / N
    
    def kernel_cross_covariance[T: (float64, float32), N: uint16, M1: uint8, M2: uint8](
        data1: "T[N, M1]",    
        data2: "T[N, M2]",    
        mean1: "T[M1]",      
        mean2: "T[M2]",      
        cov: "T[M1, M2]"     
    ):
        # Compute cross-covariance
        for i_c, j_c in allo.grid(M1, M2): # outer loop: M1, M2
            covariance: T = 0.0
            for p_c in allo.grid(N): # inner loop: N
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
    # 2. Main cca kernel
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
        
        # # trans_x:[N,M1] -> [M1,N] : (1000, 9) -> (9, 1000)
        # kernel_transpose[T, N, M1, "trans_x"](X, X_T)                        
        # # trans_y:[N,M2] -> [M2,N] : (1000, 10) -> (10, 1000)
        # kernel_transpose[T, N, M2, "trans_y"](Y, Y_T)                     
        # # cov_xx:[N,M1] -> [M1,M1] : (1000, 9) -> (9, 9)
        # kernel_mean[T, N, M1, "mean_xx"](X, X_mean)
        # kernel_cross_covariance[T, N, M1, M1, "cov_xx"](X, X, X_mean, X_mean, Cxx)
        # # cov_yy:[N,M2] -> [M2,M2] : (1000, 10) -> (10, 10)
        # kernel_mean[T, N, M2, "mean_yy"](Y, Y_mean)
        # kernel_cross_covariance[T, N, M2, M2, "cov_yy"](Y, Y, Y_mean, Y_mean, Cyy) 
        # # cov_xy:[N,M1] @ [N,M2] -> [M1,M2] : (1000, 9) @ (1000, 10) -> (9, 10)
        # kernel_cross_covariance[T, N, M1, M2, "cov_xy"](X, Y, X_mean, Y_mean, Cxy) 
        # # trans_cxy:[M1,M2] -> [M2,M1] : (9, 10) -> (10, 9)
        # kernel_transpose[T, M1, M2, "trans_cxy"](Cxy, Cyx)
        # # pinv_xx:[M1,M1] -> [M1,M1] : (9, 9) -> (9, 9) 
        # kernel_pinverse[T, M1, "pinv_xx"](Cxx, Cxx_inv, temp1_M1, temp2_M1)
        # # pinv_yy:[M2,M2] -> [M2,M2] : (10, 10) -> (10, 10)
        # kernel_pinverse[T, M2, "pinv_yy"](Cyy, Cyy_inv, temp1_M2, temp2_M2) 
        # # gemm_xx_xy:[M1,M1] @ [M1,M2] -> [M1,M2] : (9, 9) @ (9, 10) -> (9, 10)
        # kernel_gemm[T, M1, M1, M2, "gemm_xx_xy"](Cxx_inv, Cxy, temp3_M1)
        # # gemm_xy_yy:[M1,M2] @ [M2,M2] -> [M1,M2] : (9, 10) @ (10, 10) -> (9, 10)
        # kernel_gemm[T, M1, M2, M2, "gemm_xy_yy"](temp3_M1, Cyy_inv, temp4_M1) 
        # # gemm_xy_yx:[M1,M2] @ [M2,M1] -> [M1,M1] : (9, 10) @ (10, 9) -> (9, 9)
        # kernel_gemm[T, M1, M2, M1, "gemm_xy_yx"](temp4_M1, Cyx, M)
        # # eigenvalue:[M1,M1] -> [M1] : (9, 9) -> (9)
        # kernel_eigenvalue[T, M1, "eigen_m"](M, eigenvals, Q, R)
        # # sqrt:[M1] -> [2] : (9) -> (2)
        # kernel_sqrt[T, M1, "sqrt"](eigenvals, r)
        # trans_x:[N,M1] -> [M1,N] : (1000, 9) -> (9, 1000)
        kernel_transpose[T, N, M1](X, X_T)                        
        # trans_y:[N,M2] -> [M2,N] : (1000, 10) -> (10, 1000)
        kernel_transpose[T, N, M2](Y, Y_T)                     
        # cov_xx:[N,M1] -> [M1,M1] : (1000, 9) -> (9, 9)
        kernel_mean[T, N, M1](X, X_mean)
        kernel_cross_covariance[T, N, M1, M1](X, X, X_mean, X_mean, Cxx)
        # cov_yy:[N,M2] -> [M2,M2] : (1000, 10) -> (10, 10)
        kernel_mean[T, N, M2](Y, Y_mean)
        kernel_cross_covariance[T, N, M2, M2](Y, Y, Y_mean, Y_mean, Cyy) 
        # cov_xy:[N,M1] @ [N,M2] -> [M1,M2] : (1000, 9) @ (1000, 10) -> (9, 10)
        kernel_cross_covariance[T, N, M1, M2](X, Y, X_mean, Y_mean, Cxy) 
        # trans_cxy:[M1,M2] -> [M2,M1] : (9, 10) -> (10, 9)
        kernel_transpose[T, M1, M2](Cxy, Cyx)
        # pinv_xx:[M1,M1] -> [M1,M1] : (9, 9) -> (9, 9) 
        kernel_pinverse[T, M1](Cxx, Cxx_inv, temp1_M1, temp2_M1)
        # pinv_yy:[M2,M2] -> [M2,M2] : (10, 10) -> (10, 10)
        kernel_pinverse[T, M2](Cyy, Cyy_inv, temp1_M2, temp2_M2) 
        # gemm_xx_xy:[M1,M1] @ [M1,M2] -> [M1,M2] : (9, 9) @ (9, 10) -> (9, 10)
        kernel_gemm[T, M1, M1, M2](Cxx_inv, Cxy, temp3_M1)
        # gemm_xy_yy:[M1,M2] @ [M2,M2] -> [M1,M2] : (9, 10) @ (10, 10) -> (9, 10)
        kernel_gemm[T, M1, M2, M2](temp3_M1, Cyy_inv, temp4_M1) 
        # gemm_xy_yx:[M1,M2] @ [M2,M1] -> [M1,M1] : (9, 10) @ (10, 9) -> (9, 9)
        kernel_gemm[T, M1, M2, M1](temp4_M1, Cyx, M)
        # eigenvalue:[M1,M1] -> [M1] : (9, 9) -> (9)
        kernel_eigenvalue[T, M1](M, eigenvals, Q, R)
        # sqrt:[M1] -> [2] : (9) -> (2)
        kernel_sqrt[T, M1](eigenvals, r)

    #---------------------------------------------------------------------------------
    # 4. Main SSVEP-CCA kernel
    #---------------------------------------------------------------------------------
    def kernel_ref_signal[T: (float64, float32), N: uint16, M2: uint8](
        tidx: "T[N]",           # Time points array 1000
        freq: T,                # Target frequency 8.0
        Y: "T[N, M2]"          # Output reference signal matrix 1000*10
    ):
        pi: T = 3.14159265359
        pi2: T = 2.0 * pi
        curr_freq: T = 0.0
        angle: T = 0.0
        sin_idx: uint8 = 0
        cos_idx: uint8 = 0
        for harm_i in allo.grid(5):
            sin_idx = harm_i * 2
            cos_idx = harm_i * 2 + 1
            for t in allo.grid(N):
                curr_freq = freq * (harm_i + 1)
                angle = pi2 * curr_freq * tidx[t]
                # Normalize angle to [-π/2, π/2] for better precision
                for _ in range(400):
                    if angle > pi2:
                        angle = angle - pi2
                if angle > pi:
                    angle = angle - pi2
                
                # Calculate sin using Taylor series
                x2: T = angle * angle
                term: T = angle
                sin_result: T = angle
                
                # 3! term
                term = term * (-x2) / (6.0)
                sin_result = sin_result + term
                
                # 5! term
                term = term * (-x2) / (20.0)
                sin_result = sin_result + term
                
                # 7! term
                term = term * (-x2) / (42.0)
                sin_result = sin_result + term
                
                # 9! term
                term = term * (-x2) / (72.0)
                sin_result = sin_result + term

                # 11! term
                term = term * (-x2) / (110.0)
                sin_result = sin_result + term
                
                # 13! term
                term = term * (-x2) / (156.0)
                sin_result = sin_result + term
                
                # Store sin value
                Y[t, sin_idx] = sin_result
                
                # Calculate cos: shift angle by π/2
                angle = angle + pi/2.0
                for _ in range(400):
                    if angle > pi2:
                        angle = angle - pi2
                if angle > pi:
                    angle = angle - pi2
                
                x2 = angle * angle
                term = angle
                cos_result: T = angle
                
                # 3! term
                term = term * (-x2) / (6.0)
                cos_result = cos_result + term
                
                # 5! term
                term = term * (-x2) / (20.0)
                cos_result = cos_result + term
                
                # 7! term
                term = term * (-x2) / (42.0)
                cos_result = cos_result + term

                # 9! term
                term = term * (-x2) / (72.0)
                cos_result = cos_result + term

                # 11! term
                term = term * (-x2) / (110.0)
                cos_result = cos_result + term

                # 13! term  
                term = term * (-x2) / (156.0)
                cos_result = cos_result + term
                
                # Store cos value
                Y[t, cos_idx] = cos_result
                Y[t, cos_idx] = cos_result

    #---------------------------------------------------------------------------------
    # 4. Main SSVEP-CCA kernel
    #---------------------------------------------------------------------------------
    def kernel_ssvep_cca[T: (float64, float32), N: uint16, M1: uint8, M2: uint8, F: uint8](
        X: "T[N, M1]",          # EEG signal 1000*9
        freqs: "T[F]",          # frequency array 40
        max_r: "T[2]",          # maximum correlation coefficient
        max_frq: "T[2]"         # maximum correlation coefficient corresponding frequency index
    ):
        # temp
        Y: "T[N, M2]"          # reference signal matrix 1000*10
        r: "T[2]"              # current correlation coefficient
        tidx: "T[N]"           # time point array 1000
        curr_freq: T
        # initialize maximum correlation coefficient
        max_r[0] = -1.0
        max_r[1] = 0.0
        max_frq[0] = 0.0
        max_frq[1] = 0.0

        # generate time sequence: tidx = np.arange(1, N + 1) / sampling_rate
        for t in allo.grid(N):
            tidx[t] = (t + 1) * 1.0 * 4 / N
        
        # iterate all frequencies
        for freq_idx in allo.grid(F):
            curr_freq = freqs[freq_idx]
            # generate reference signal
            kernel_ref_signal[T, N, M2](tidx, curr_freq, Y)
            # compute CCA with EEG signal X and reference signal Y
            kernel_cca[T, N, M1, M2](X, Y, r)
            if r[0] > max_r[0]:
                max_r[0] = r[0]
                max_r[1] = 0.0
                max_frq[0] = curr_freq
                max_frq[1] = 0.0

    # ---------------------------------------------------------------------------------  
    # 5. Optimize and compose CCA kernels
    # ---------------------------------------------------------------------------------  

    # #---------------------------------------------------------------------------------  
    # # Transpose kernel
    # #---------------------------------------------------------------------------------
    # s1  = allo.customize(kernel_transpose, instantiate=[concrete_type, N, M1])      # [N,M1] -> [M1,N] : (1000, 9) -> (9, 1000)
    # s2  = allo.customize(kernel_transpose, instantiate=[concrete_type, N, M2])      # [N,M2] -> [M2,N] : (1000, 10) -> (10, 1000)   
    # s3  = allo.customize(kernel_transpose, instantiate=[concrete_type, M1, M2])     # [M1,M2] -> [M2,M1] : (9, 10) -> (10, 9)

    # #---------------------------------------------------------------------------------  
    # # Covariance kernel
    # #---------------------------------------------------------------------------------
    # s4  = allo.customize(kernel_mean, instantiate=[concrete_type, N, M1]) # [N,M1] -> [M1] : (1000, 9) -> (9)
    # s5  = allo.customize(kernel_cross_covariance, instantiate=[concrete_type, N, M1, M1]) # [N,M1] @ [M1,N] -> [M1,M1] : (1000, 9) @ (1000, 9) -> (9, 9)
    
    # s6  = allo.customize(kernel_mean, instantiate=[concrete_type, N, M2]) # [N,M2] -> [M2] : (1000, 10) -> (10)
    # s7  = allo.customize(kernel_cross_covariance, instantiate=[concrete_type, N, M2, M2]) # [N,M2] @ [M2,N] -> [M2,M2] : (1000, 10) @ (1000, 10) -> (10, 10)  
    
    # s8  = allo.customize(kernel_cross_covariance, instantiate=[concrete_type, N, M1, M2]) # [N,M1] @ [M2,N] -> [M1,M2] : (1000, 9) @ (1000, 10) -> (9, 10)

    # #---------------------------------------------------------------------------------  
    # # Pseudo-inverse kernel
    # #---------------------------------------------------------------------------------
    # s9  = allo.customize(kernel_pinverse, instantiate=[concrete_type, M1])          # [M1,M1] @ [M1,M1] -> [M1,M1] : (9, 9) @ (9, 9) -> (9, 9)
    # s10 = allo.customize(kernel_pinverse, instantiate=[concrete_type, M2])          # [M2,M2] @ [M2,M2] -> [M2,M2] : (10, 10) @ (10, 10) -> (10, 10)

    # #---------------------------------------------------------------------------------  
    # # GEMM kernel
    # #---------------------------------------------------------------------------------
    # s11 = allo.customize(kernel_gemm, instantiate=[concrete_type, M1, M1, M2])      # [M1,M1] @ [M1,M2] -> [M1,M2] : (9, 9) @ (9, 10) -> (9, 10)
    # s12 = allo.customize(kernel_gemm, instantiate=[concrete_type, M1, M2, M2])      # [M1,M2] @ [M2,M2] -> [M1,M2] : (9, 10) @ (10, 10) -> (9, 10)  
    # s13 = allo.customize(kernel_gemm, instantiate=[concrete_type, M1, M2, M1])      # [M1,M2] @ [M2,M1] -> [M1,M1] : (9, 10) @ (10, 9) -> (9, 9)

    # #---------------------------------------------------------------------------------  
    # # Eigenvalue kernel
    # #---------------------------------------------------------------------------------
    # s14 = allo.customize(kernel_eigenvalue, instantiate=[concrete_type, M1])        # [M1,M1] @ [M1,M1] -> [M1] : (9, 9) @ (9, 9) -> (9)

    # #---------------------------------------------------------------------------------  
    # # Square root kernel
    # #---------------------------------------------------------------------------------
    # s15 = allo.customize(kernel_sqrt, instantiate=[concrete_type, M1])              # [M1] -> [2] : (9) -> (2)

    # #---------------------------------------------------------------------------------  
    # # Reference signal generation kernel
    # #---------------------------------------------------------------------------------
    # s16 = allo.customize(kernel_ref_signal, instantiate=[concrete_type, N, M2])

    # #---------------------------------------------------------------------------------  
    # # Compose kernels
    # #---------------------------------------------------------------------------------
    # # First compose the CCA kernel
    # sch1 = allo.customize(kernel_cca, instantiate=[concrete_type, N, M1, M2])

    # # Compose transpose kernels
    # sch1.compose(s1, id="trans_x")
    # sch1.compose(s2, id="trans_y")
    # sch1.compose(s3, id="trans_cxy")

    # # Compose mean and covariance kernels
    # sch1.compose(s4, id="mean_xx")
    # sch1.compose(s5, id="cov_xx")
    # sch1.compose(s6, id="mean_yy")
    # sch1.compose(s7, id="cov_yy")
    # sch1.compose(s8, id="cov_xy")

    # # Compose matrix operation kernels
    # sch1.compose(s9, id="pinv_xx")
    # sch1.compose(s10, id="pinv_yy")
    # sch1.compose(s11, id="gemm_xx_xy")
    # sch1.compose(s12, id="gemm_xy_yy")
    # sch1.compose(s13, id="gemm_xy_yx")
    # sch1.compose(s14, id="eigen_m")
    # sch1.compose(s15, id="sqrt")

    # # Then compose the SSVEP-CCA kernel
    sch = allo.customize(kernel_ssvep_cca, instantiate=[concrete_type, N, M1, M2, F])

    # # Compose reference signal generation and CCA kernels
    # sch.compose(s16, id="ref_gen")
    # sch.compose(sch1, id="cca")

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
# Test SSVEP-CCA algorithm
#================================================================================
#---------------------------------------------------------------------------------
# Define parameters
#---------------------------------------------------------------------------------
sampling_rate = 250  # Sampling rate in Hz
gaze_length = 4     # Gaze time in seconds
N = 1000 # number of samples
M1 = 9             # EEG channels
num_harmonics = 5  # Number of harmonics for reference signals
M2 = 2 * num_harmonics  # Reference signals (sine and cosine for each harmonic)
F = 40 # Number of frequencies
list_freqs = np.round(np.arange(8, 16, 0.2), decimals=1).astype(np.float32)

#---------------------------------------------------------------------------------
# Define target and block parameters
#---------------------------------------------------------------------------------
target_idx = 12  # Target number (1-40)
block_idx = 1   # Block number (1-6)

print(f"\nSelected target parameters:")
print(f"Block number: {block_idx}")
print(f"Target number: {target_idx}")
print(f"Sampling rate: {sampling_rate} Hz")
print(f"Gaze length: {gaze_length} seconds")
print(f"Number of harmonics: {num_harmonics}")
# print(f"Frequency range: {list_freqs}")

# Build the file path
base_dir = "/home/sx286/allo/BCI/test/EEG_Benchmark/extracted_data"
eeg_data_path = os.path.join(base_dir, f"S2_target_{target_idx}_block_{block_idx}.npy")

# Check if the file exists
if not os.path.exists(eeg_data_path):
    raise FileNotFoundError(f"EEG data file not found: {eeg_data_path}")

# Load extracted EEG data
eeg_data = np.load(eeg_data_path)
print(f"Loaded EEG data shape: {eeg_data.shape}")

X = eeg_data.astype(np.float32)

#---------------------------------------------------------------------------------
# CCA algorithm schedule
#---------------------------------------------------------------------------------

concrete_type = float32
sch = cca_algorithm(concrete_type, N, M1, M2, F)

#---------------------------------------------------------------------------------
# HLS
#---------------------------------------------------------------------------------
# mod = sch.build()
# mod = sch.build(target="vitis_hls",mode="csim",project="cca_t12_415_2236.prj")
mod = sch.build(target="vitis_hls",mode="hw_emu",project="cca_t12_415_2236.prj")
# mod = sch.build(target="vitis_hls",mode="hw",project="cca_t3_415_2106.prj")

#---------------------------------------------------------------------------------
# Initialize outputs
#---------------------------------------------------------------------------------
max_r = np.zeros(2, dtype=np.float32)
max_frq = np.zeros(2, dtype=np.float32)  # 修改为float32[2]

# Run SSVEP-CCA
mod(X, list_freqs, max_r, max_frq)

# Target frequency mapping (get the corresponding frequency according to the target number)
ref_freqs = {
    1: 8.0,   2: 9.0,   3: 10.0,  4: 11.0,  5: 12.0,  6: 13.0,  7: 14.0,  8: 15.0,
    9: 8.2,   10: 9.2,  11: 10.2, 12: 11.2, 13: 12.2, 14: 13.2, 15: 14.2, 16: 15.2,
    17: 8.4,  18: 9.4,  19: 10.4, 20: 11.4, 21: 12.4, 22: 13.4, 23: 14.4, 24: 15.4,
    25: 8.6,  26: 9.6,  27: 10.6, 28: 11.6, 29: 12.6, 30: 13.6, 31: 14.6, 32: 15.6,
    33: 8.8,  34: 9.8,  35: 10.8, 36: 11.8, 37: 12.8, 38: 13.8, 39: 14.8, 40: 15.8
}

# Print results
print("\n==== SSVEP-CCA Results ====")
print(f"max_r: {max_r[0]}")
print(f"max_frq: {max_frq[0]}")
print(f"Detected frequency: {max_frq[0]}", "Expected frequency: ", ref_freqs[target_idx])

def find_key_by_value(ref_dict, target_value):
    for key, value in ref_dict.items():
        if value == target_value:
            return key
    return None 

target_value = max_frq[0]
found_key = find_key_by_value(ref_freqs, target_value)
print(f"Detected target: {found_key}", "Expected target: ", target_idx)
if found_key == target_idx:
    print("Target detection successful!")
else:
    print("Target detection failed!")

# #---------------------------------------------------------------------------------
# # Test all targets
# #---------------------------------------------------------------------------------
# total_correct = 0
# total_trials = 0

# print("\n==== Testing All Targets ====")
# print("Sampling rate: {}Hz, Gaze length: {}s, Number of harmonics: {}\n".format(sampling_rate, gaze_length, num_harmonics))

# for target_idx in range(1, 41):  # 1-40
#     # Build the file path
#     eeg_data_path = os.path.join(base_dir, f"S2_target_{target_idx}_block_{block_idx}.npy")
    
#     try:
#         # Load extracted EEG data
#         eeg_data = np.load(eeg_data_path)
#         X = eeg_data.astype(np.float32)
        
#         # Initialize outputs
#         max_r = np.zeros(2, dtype=np.float32)
#         max_frq = np.zeros(2, dtype=np.float32)
        
#         # Run SSVEP-CCA
#         mod(X, list_freqs, max_r, max_frq)
        
#         # Find detected target
#         detected_freq = max_frq[0]
#         detected_target = find_key_by_value(ref_freqs, detected_freq)
        
#         # Check if detection was correct
#         is_correct = detected_target == target_idx
#         if is_correct:
#             total_correct += 1
#         total_trials += 1
        
#         # Print results
#         print(f"Target {target_idx:2d} (Expected frequency: {ref_freqs[target_idx]:4.1f}Hz):")
#         print(f"     Detected target: {detected_target:2d}, Detected frequency: {detected_freq:4.1f}Hz")
#         print(f"     Correlation coefficient: {max_r[0]:.4f}")
#         print(f"     Result: {'✓' if is_correct else '✗'}\n")
        
#     except FileNotFoundError:
#         print(f"Warning: Data file for target {target_idx} does not exist, skipping...\n")
#         continue

# # Print final accuracy
# accuracy = (total_correct / total_trials) * 100 if total_trials > 0 else 0
# print("\n==== Final Results ====")
# print(f"Total trials: {total_trials}")
# print(f"Correctly identified: {total_correct}")
# print(f"Accuracy: {accuracy:.2f}%")
