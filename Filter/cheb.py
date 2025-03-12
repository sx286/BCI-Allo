# ------------------------------------------------------------------------------
# Chebyshev filter design using scipy.signal
# ------------------------------------------------------------------------------
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest
import numpy as np
import allo
import allo.ir.types as T
from allo.ir.types import float32, float64, int32
from scipy import signal

#----------------------------------------------------------------
# Python implementation using scipy.signal
#----------------------------------------------------------------
def design_cheby1_filter_py(wp, ws, gpass, gstop):
    """
    Design Type I Chebyshev bandpass filter
    
    Parameters:
    wp: passband frequencies [wp1, wp2]
    ws: stopband frequencies [ws1, ws2]
    gpass: passband ripple (dB)
    gstop: stopband attenuation (dB)
    
    Returns:
    N: filter order
    Wn: passband cutoff frequencies
    B: numerator coefficients
    A: denominator coefficients
    """
    # Calculate filter order and cutoff frequencies
    N, Wn = signal.cheb1ord(wp, ws, gpass, gstop)
    
    # Design filter
    B, A = signal.cheby1(N, 0.5, Wn, btype='bandpass')
    
    return N, Wn, B, A

#----------------------------------------------------------------
# Allo implementation of Chebyshev filter design
#----------------------------------------------------------------
def design_cheby1_filter(type):
    """
    Complete Allo implementation of Type I Chebyshev filter design
    """
    def kernel_cheby1[T: (float32, float64)](
        wp: "T[2]",          # Passband frequencies [wp1, wp2]
        ws: "T[2]",          # Stopband frequencies [ws1, ws2]
        gpass: T,            # Passband ripple (dB)
        gstop: T,            # Stopband attenuation (dB)
        b: "T[17]",          # Output: numerator coefficients
        a: "T[17]"           # Output: denominator coefficients
    ):
        # Helper function: Calculate sin using Taylor series
        def sin_approx(x: T) -> T:
            x2: T = x * x
            x3: T = x2 * x
            x5: T = x3 * x2
            x7: T = x5 * x2
            return x - x3/6.0 + x5/120.0 - x7/5040.0

        # Helper function: Calculate cos using Taylor series
        def cos_approx(x: T) -> T:
            x2: T = x * x
            x4: T = x2 * x2
            x6: T = x4 * x2
            return 1.0 - x2/2.0 + x4/24.0 - x6/720.0

        # 1. Calculate filter order
        eps: T = (10.0 ** (0.1 * gpass) - 1.0) ** 0.5
        eps_s: T = (10.0 ** (0.1 * gstop) - 1.0) ** 0.5
        
        # Calculate frequency pre-warping
        wp_c: T = (wp[1] + wp[0]) / 2.0  # Center frequency
        bw: T = wp[1] - wp[0]            # Bandwidth
        
        # 2. Calculate Chebyshev polynomial coefficients
        def compute_cheby_poly(x: T, n: int32) -> T:
            if n == 0:
                return 1.0
            elif n == 1:
                return x
            else:
                t_prev: T = 1.0
                t_curr: T = x
                t_next: T = 0.0
                
                for i in range(2, n + 1):
                    t_next = 2.0 * x * t_curr - t_prev
                    t_prev = t_curr
                    t_curr = t_next
                
                return t_curr
        
        # 3. Calculate poles of analog prototype lowpass filter
        n: int32 = 8  # Filter order
        
        # Initialize coefficient arrays
        for i in allo.grid(17):
            b[i] = 0.0
            a[i] = 0.0
            
        # Calculate prototype lowpass filter coefficients
        for k in allo.grid(n):
            theta: T = (2.0 * k + 1.0) * 3.14159265359 / (2.0 * n)
            
            # Calculate poles
            re: T = -eps * sin_approx(theta)
            im: T = eps * cos_approx(theta)
            
            # Update filter coefficients
            b[k] = re
            a[k] = im
        
        # 4. Frequency transformation (lowpass to bandpass)
        w0: T = 2.0 * 3.14159265359 * wp_c
        alpha: T = sin_approx(w0) / (2.0 * bw)
        
        # Temporary arrays to store original coefficients
        b_temp: "T[17]"
        a_temp: "T[17]"
        for i in allo.grid(17):
            b_temp[i] = b[i]
            a_temp[i] = a[i]
        
        # Bandpass transformation
        for i in allo.grid(2*n+1):
            if i == 0:
                a[i] = 1.0
            else:
                if i % 2 == 0:
                    b[i] = -b_temp[i-1]
                    a[i] = -a_temp[i-1]
                else:
                    b[i] = 2.0 * alpha * b_temp[i-1]
                    a[i] = 2.0 * alpha * a_temp[i-1]
        
        # Normalize coefficients
        a0: T = a[0]
        if a0 != 0.0:
            for i in allo.grid(17):
                b[i] = b[i] / a0
                a[i] = a[i] / a0

    s = allo.customize(kernel_cheby1, instantiate=[type])
    return s.build()

#----------------------------------------------------------------
# Test implementation
#----------------------------------------------------------------
def test_cheby1_filter():
    """
    Test Chebyshev filter design
    """
    # Test parameters
    fs = 250.0  # Sampling frequency
    wp = np.array([10/125, 90/125])  # Passband frequencies
    ws = np.array([8/125, 100/125])  # Stopband frequencies
    gpass = 3.0   # Passband ripple
    gstop = 40.0  # Stopband attenuation
    
    # Python implementation
    N_py, Wn_py, B_py, A_py = design_cheby1_filter_py(wp, ws, gpass, gstop)
    
    # Initialize Allo output arrays
    b = np.zeros(17, dtype=np.float64)
    a = np.zeros(17, dtype=np.float64)
    
    # Allo implementation
    mod = design_cheby1_filter(float64)
    mod(wp, ws, gpass, gstop, b, a)
    
    # Print comparison results
    print("\nFilter Design Results Comparison:")
    print(f"Filter Order N:")
    print(f"Python implementation: {N_py}")
    print(f"Allo implementation: {len(b) - 1}")
    
    print(f"\nCutoff Frequencies Wn:")
    print(f"Python implementation: {Wn_py}")
    print(f"Allo implementation: {np.sqrt(b[1] * b[1] + a[1] * a[1])}")
    
    print(f"\nNumerator Coefficients B:")
    print(f"Python implementation: {B_py}")
    print(f"Allo implementation (non-zero part): {b[:len(B_py)]}")
    
    print(f"\nDenominator Coefficients A:")
    print(f"Python implementation: {A_py}")
    print(f"Allo implementation (non-zero part): {a[:len(A_py)]}")
    
    # Verify results
    assert len(b) - 1 == N_py, "Filter order mismatch"
    np.testing.assert_allclose(np.sqrt(b[1] * b[1] + a[1] * a[1]), Wn_py, rtol=1e-10, atol=1e-10, 
                             err_msg="Cutoff frequency mismatch")
    np.testing.assert_allclose(b[:len(B_py)], B_py, rtol=1e-10, atol=1e-10,
                             err_msg="Numerator coefficients mismatch")
    np.testing.assert_allclose(a[:len(A_py)], A_py, rtol=1e-10, atol=1e-10,
                             err_msg="Denominator coefficients mismatch")

if __name__ == "__main__":
    pytest.main([__file__])
