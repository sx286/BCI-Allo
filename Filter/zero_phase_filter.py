# ------------------------------------------------------------------------------
# Zero-phase filtering for multi-channel signals
# ------------------------------------------------------------------------------
# Copyright Allo authors. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

import os
import json
import pytest
import numpy as np
import allo
import allo.ir.types as T
from allo.ir.types import float32, float64, int32, uint8, uint16
from scipy import signal

#----------------------------------------------------------------
# Numpy implementation of zero-phase filtering
#----------------------------------------------------------------
def zero_phase_filter_py(data, b, a, filtered, num_channels, num_samples):
    """
    Apply zero-phase filtering to multi-channel signals
    
    Parameters:
    data: input signal (num_channels, num_samples)
    b: numerator coefficients of the filter
    a: denominator coefficients of the filter
    filtered: output filtered signal
    num_channels: number of channels (uint8, ≤ 255)
    num_samples: number of samples (uint16, ≤ 65535)
    """
    # Process each channel
    for ch in range(num_channels):
        # Apply zero-phase filter using scipy's filtfilt
        filtered[ch, :] = signal.filtfilt(b, a, data[ch, :])

#----------------------------------------------------------------
# Allo implementation of zero-phase filtering
#----------------------------------------------------------------
def zero_phase_filter(type, num_channels, num_samples):
    """
    Allo implementation of zero-phase filtering
    
    Signal is 2D array (num_channels, num_samples)
    Signal's amplitude is float64 or float32
    num_channels is uint8 (max 255 channels)
    num_samples is uint16 (max 65535 samples)
    """
    def kernel_filter[T: (float32, float64), C: int32, N: int32](
        data: "T[C, N]",      # Input signal
        filtered: "T[C, N]"
        ,   # Output filtered signal
        b: "T[17]",           # Filter numerator coefficients
        a: "T[17]"            # Filter denominator coefficients
    ):
        # Initialize temporary array
        temp: "T[C, N]"
        for c, n in allo.grid(C, N):
            temp[c, n] = 0.0
            filtered[c, n] = 0.0
        
        # Process each channel
        for ch in allo.grid(C):
            # Forward filtering
            x_prev: "T[16]"  # Previous input values
            y_prev: "T[16]"  # Previous output values
            
            # Initialize delay lines
            for i in allo.grid(16):
                x_prev[i] = 0.0
                y_prev[i] = 0.0
            
            # Forward pass
            for i in allo.grid(N):
                # Get current input
                x_curr: T = data[ch, i]
                
                # Compute output
                y_curr: T = b[0] * x_curr
                
                # Add previous inputs (b coefficients)
                for j in allo.grid(16):
                    y_curr = y_curr + b[j+1] * x_prev[j]
                
                # Subtract previous outputs (a coefficients)
                for j in allo.grid(16):
                    y_curr = y_curr - a[j+1] * y_prev[j]
                
                # Store result
                temp[ch, i] = y_curr
                
                # Update delay lines
                for j in allo.grid(15):
                    x_prev[15-j] = x_prev[14-j]
                    y_prev[15-j] = y_prev[14-j]
                x_prev[0] = x_curr
                y_prev[0] = y_curr
            
            # Backward filtering
            # Reset delay lines
            for i in allo.grid(16):
                x_prev[i] = 0.0
                y_prev[i] = 0.0
            
            # Backward pass
            for i in allo.grid(N):
                idx: int32 = N - 1 - i
                # Get current input
                x_curr: T = temp[ch, idx]
                
                # Compute output
                y_curr: T = b[0] * x_curr
                
                # Add previous inputs (b coefficients)
                for j in allo.grid(16):
                    y_curr = y_curr + b[j+1] * x_prev[j]
                
                # Subtract previous outputs (a coefficients)
                for j in allo.grid(16):
                    y_curr = y_curr - a[j+1] * y_prev[j]
                
                # Store result
                filtered[ch, idx] = y_curr
                
                # Update delay lines
                for j in allo.grid(15):
                    x_prev[15-j] = x_prev[14-j]
                    y_prev[15-j] = y_prev[14-j]
                x_prev[0] = x_curr
                y_prev[0] = y_curr

    s = allo.customize(kernel_filter, instantiate=[type, num_channels, num_samples])
    return s.build()

#----------------------------------------------------------------
# Test filter implementation against numpy implementation
#----------------------------------------------------------------
def test_zero_phase_filter():
    """
    Test zero-phase filter implementation against numpy implementation
    """
    # Test parameters
    num_channels = 64    # Number of channels
    num_samples = 1000   # Number of samples
    fs = 250            # Sampling frequency (Hz)
    
    # Filter design parameters
    Wp = [10/125, 90/125]  # Normalized passband frequencies
    Ws = [8/125, 100/125]  # Normalized stopband frequencies
    gpass = 3              # Passband ripple (dB)
    gstop = 40            # Stopband attenuation (dB)
    
    # Design filter
    order, wn = signal.cheb1ord(Wp, Ws, gpass, gstop)
    print(f"Filter order: {order}")
    
    # Design filter
    b, a = signal.cheby1(order, 0.5, wn, btype='bandpass')
    print(f"Filter coefficient length: b={len(b)}, a={len(a)}")
    print(f"Numerator coefficients b: {b}")
    print(f"Denominator coefficients a: {a}")
    
    # Print non-zero b coefficients indices
    non_zero_b = np.nonzero(b)[0]
    print(f"Non-zero b coefficients indices: {non_zero_b}")
    print(f"Non-zero b coefficients values: {b[non_zero_b]}")
    
    # Ensure coefficients are length coeff_len
    coeff_len = 2 * order + 1
    print(f"Required coefficient length: {coeff_len}")
    
    # Generate test signal
    t = np.linspace(0, (num_samples-1)/fs, num_samples)
    data = np.zeros((num_channels, num_samples))
    for ch in range(num_channels):
        freq = 5 + ch * 2  # Different frequency for each channel
        data[ch, :] = np.sin(2*np.pi*freq*t)
    data = data.astype(np.float64)
    
    # Initialize output arrays
    filtered = np.zeros_like(data)
    filtered_ref = np.zeros_like(data)
    
    # Apply numpy implementation
    zero_filter_np = np.zeros_like(data)
    for ch in range(num_channels):
        # Forward filter
        zero_filter_np[ch, :] = signal.lfilter(b, a, data[ch, :])
        # Backward filter
        zero_filter_np[ch, :] = signal.lfilter(b, a, zero_filter_np[ch, ::-1])[::-1]
    filtered_ref = zero_filter_np
    
    # Apply Allo implementation
    mod = zero_phase_filter(float64, num_channels, num_samples)
    mod(data, filtered, b, a)
    
    # Print some debug values
    print("\n forward filter result comparison (first 10 samples of the first channel):")
    print("Allo implementation:", filtered[0, :10])
    print("NumPy implementation:", filtered_ref[0, :10])
    print("\nDifference:", filtered[0, :10] - filtered_ref[0, :10])
    print("Relative difference:", np.abs((filtered[0, :10] - filtered_ref[0, :10])/filtered_ref[0, :10]))
    
    # Verify results
    np.testing.assert_allclose(filtered, filtered_ref, rtol=1e-5, atol=1e-5)

if __name__ == "__main__":
    pytest.main([__file__])
