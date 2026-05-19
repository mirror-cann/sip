#!/usr/bin/env python3
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

DTYPE_INT_TO_NAME = {
    16: "complex64",
    33: "complex32",
}

NAME_TO_NP_TYPE = {
    "complex64": np.complex64,
    "complex32": None,  # numpy 没有 complex32，用 float16 表示
}


def load_tensor_from_bin(filename):
    with open(filename, "rb") as f:
        dtype_int = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dtype_name = DTYPE_INT_TO_NAME.get(dtype_int)
        if dtype_name is None:
            raise ValueError(f"Unsupported dtype: {dtype_int}")

        dim_count = np.frombuffer(f.read(4), dtype=np.int32)[0]
        dims = np.frombuffer(f.read(8 * dim_count), dtype=np.int64)
        raw_data = f.read()

        if dtype_name == "complex64":
            data = np.frombuffer(raw_data, dtype=np.complex64).reshape(dims)
        elif dtype_name == "complex32":
            # complex32 = 2 * float16 per element
            raw_f16 = np.frombuffer(raw_data, dtype=np.float16)
            n = dims[0]
            data = np.zeros(n, dtype=np.complex64)
            for i in range(n):
                data[i] = complex(float(raw_f16[2 * i]), float(raw_f16[2 * i + 1]))
        return data, dtype_name


def gen_golden_data(dtype):
    if dtype == "c64":
        input_files = ["complex64_input_0.bin", "complex64_input_1.bin"]
        np_type = np.complex64
        dtype_name = "complex64"
    elif dtype == "c32":
        input_files = ["complex32_input_0.bin", "complex32_input_1.bin"]
        np_type = np.complex64  # 计算用 complex64
        dtype_name = "complex32"
    else:
        raise ValueError(f"Unknown dtype: {dtype}")

    data_x, _ = load_tensor_from_bin(os.path.join(curr_dir, input_files[0]))
    data_y, _ = load_tensor_from_bin(os.path.join(curr_dir, input_files[1]))

    golden = data_x * data_y

    if dtype_name == "complex64":
        golden.astype(np.complex64).tofile(
            os.path.join(curr_dir, "complex64_golden_t_asd_mul_0.bin")
        )
        print(f"Generated: complex64_golden_t_asd_mul_0.bin")
    else:
        # complex32: 保存为 float16 格式
        golden_f16 = np.zeros(len(golden) * 2, dtype=np.float16)
        for i in range(len(golden)):
            golden_f16[2 * i] = np.float16(golden[i].real)
            golden_f16[2 * i + 1] = np.float16(golden[i].imag)
        golden_f16.tofile(os.path.join(curr_dir, "complex32_golden_t_asd_mul_0.bin"))
        print(f"Generated: complex32_golden_t_asd_mul_0.bin")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate golden data for asdMul")
    parser.add_argument(
        "--dtype",
        type=str,
        required=True,
        choices=["c64", "c32"],
        help="Data type: c64 or c32",
    )
    args = parser.parse_args()
    gen_golden_data(args.dtype)
