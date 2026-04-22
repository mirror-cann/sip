#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# -----------------------------------------------------------------------------------------------------------
# Copyright (c) 2026 Huawei Technologies Co., Ltd.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------

import sys
import os
import numpy as np
import argparse

curr_dir = os.path.dirname(os.path.realpath(__file__))

def load_tensor_from_bin(filename):
    """Load tensor data from binary file saved by C++ test.
    Format: dtype(int32) + dim_count(int32) + dims(int64[]) + raw_data
    """
    dtype_map = {0: 'undefined', 1: np.float32, 2: np.float16, 3: np.int32, 4: np.int64, 16: np.complex64, 6: np.uint8}
    
    with open(filename, 'rb') as f:
        # Read dtype
        dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
        np_type = dtype_map.get(dtype_int, np.float32)
        
        # Read dimension count
        dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
        
        # Read dimensions
        dims = np.frombuffer(f.read(8 * dim_count), dtype=np.int64)
        
        # Read raw data
        data = np.frombuffer(f.read(), dtype=np_type)
        data = data.reshape(dims)
        
        return data, dims, np_type

def compute_fft_c2r_golden(input_data, nfft):
    """Compute FFT C2R (Hermitian FFT) golden reference using numpy.
    
    For C2R Hermitian FFT, the input is a complex array of size N/2+1 representing 
    the first half of a Hermitian-symmetric spectrum.
    The output is a real array of size N.
    
    This is equivalent to numpy.fft.hfft(input, n=nfft)
    """
    # Use numpy's hfft which computes the FFT of Hermitian-symmetric input
    # hfft is the inverse of rfft, uses exp(-jθ) sign convention
    golden = np.fft.hfft(input_data, n=nfft)
    return golden.astype(np.float32)

def generate_golden_data(batch, nfft):
    """Load input data and generate golden reference."""
    input_filename = curr_dir + "/complex64_input_0.bin"
    
    if not os.path.exists(input_filename):
        print(f"Input file not found: {input_filename}")
        sys.exit(1)
    
    input_data, dims, dtype = load_tensor_from_bin(input_filename)
    
    print(f"Loaded input: shape={dims}, dtype={dtype}")
    print(f"Input data shape: {input_data.shape}")
    
    # Compute golden for each batch
    in_signal = nfft // 2 + 1
    golden_data = np.zeros((batch, nfft), dtype=np.float32)
    
    for b in range(batch):
        batch_input = input_data[b, :]
        golden_data[b, :] = compute_fft_c2r_golden(batch_input, nfft)
    
    # Save golden data
    golden_filename = curr_dir + "/float_golden_c2r.bin"
    golden_data.astype(np.float32).tofile(golden_filename)
    print(f"Golden data saved: {golden_filename}, shape={golden_data.shape}")
    
    # Also save input in simple format for comparison
    input_simple_filename = curr_dir + "/complex64_input_simple.bin"
    input_data.astype(np.complex64).tofile(input_simple_filename)
    
    return golden_data

def main():
    parser = argparse.ArgumentParser(description='Generate golden data for FFT C2R test')
    parser.add_argument('--batch', type=int, default=1, help='Batch size')
    parser.add_argument('--nfft', type=int, default=16, help='FFT size')
    args = parser.parse_args()
    
    print(f"Generating golden data: batch={args.batch}, nfft={args.nfft}")
    golden_data = generate_golden_data(args.batch, args.nfft)
    print(f"golden_data = {golden_data.flatten()[:10]}")
    print("Done!")

if __name__ == "__main__":
    main()