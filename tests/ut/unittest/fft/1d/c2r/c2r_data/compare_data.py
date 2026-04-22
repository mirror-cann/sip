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

def compare_data(batch, nfft, rtol=1e-3, atol=1e-5):
    """Compare NPU output with golden reference."""
    golden_file = curr_dir + "/float_golden_c2r.bin"
    output_file = curr_dir + "/float_output_.bin"
    
    if not os.path.exists(golden_file):
        print(f"Golden file not found: {golden_file}")
        sys.exit(1)
    
    if not os.path.exists(output_file):
        print(f"Output file not found: {output_file}")
        sys.exit(1)
    
    # Load data
    golden = np.fromfile(golden_file, dtype=np.float32)
    output = np.fromfile(output_file, dtype=np.float32)
    
    expected_size = batch * nfft
    print(f"Expected size: {expected_size}")
    print(f"Golden size: {golden.size}")
    print(f"Output size: {output.size}")
    
    if golden.size != expected_size or output.size != expected_size:
        print(f"ERROR: Size mismatch! Expected {expected_size}")
        sys.exit(1)
    
    golden = golden.reshape(batch, nfft)
    output = output.reshape(batch, nfft)
    
    # Compare with tolerance
    diff_res = np.isclose(output, golden, rtol=rtol, atol=atol)
    diff_idx = np.where(diff_res != True)[0]
    
    if len(diff_idx) == 0:
        print("PASSED! All values match within tolerance.")
        print(f"Max absolute error: {np.max(np.abs(output - golden))}")
        print(f"Max relative error: {np.max(np.abs(output - golden) / (np.abs(golden) + 1e-10))}")
        sys.exit(0)
    else:
        print("FAILED! Values mismatch:")
        total_errors = len(diff_idx)
        print(f"Total mismatches: {total_errors} / {expected_size} ({100*total_errors/expected_size:.2f}%)")
        
        # Show first 10 mismatches
        for idx in diff_idx[:min(10, len(diff_idx))]:
            b = idx // nfft
            n = idx % nfft
            print(f"  batch={b}, idx={n}: output={output[b, n]:.6f}, golden={golden[b, n]:.6f}, diff={output[b, n] - golden[b, n]:.6f}")
        
        sys.exit(1)

def main():
    parser = argparse.ArgumentParser(description='Compare NPU output with golden reference for FFT C2R')
    parser.add_argument('--batch', type=int, required=True, help='Batch size')
    parser.add_argument('--nfft', type=int, required=True, help='FFT size')
    parser.add_argument('--rtol', type=float, default=1e-3, help='Relative tolerance')
    parser.add_argument('--atol', type=float, default=1e-5, help='Absolute tolerance')
    args = parser.parse_args()
    
    print(f"Comparing data: batch={args.batch}, nfft={args.nfft}")
    compare_data(args.batch, args.nfft, args.rtol, args.atol)

if __name__ == '__main__':
    main()