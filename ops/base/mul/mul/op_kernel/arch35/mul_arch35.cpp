/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#define CATLASS_ARCH 3510

#include "kernel_operator.h"
#include "mul_arch35.h"

constexpr int64_t DTYPE_C64 = 0;
constexpr int64_t DTYPE_C32 = 1;

using namespace MulArch35;

extern "C" __global__ __aicore__ void mul_arch35(
    GM_ADDR gm_x, GM_ADDR gm_y, GM_ADDR gm_output, GM_ADDR wksp, GM_ADDR gm_tiling)
{
    auto tiling = reinterpret_cast<__gm__ uint8_t *>(gm_tiling);
    int64_t dtype = (*(__gm__ int64_t *)(tiling + sizeof(int64_t)));

    if (dtype == DTYPE_C64) {
        MulArch35Kernel<float> kernel;
        kernel.Init((__gm__ float *)gm_x, (__gm__ float *)gm_y, (__gm__ float *)gm_output, tiling);
        kernel.Process();
    } else {
        MulArch35Kernel<half> kernel;
        kernel.Init((__gm__ half *)gm_x, (__gm__ half *)gm_y, (__gm__ half *)gm_output, tiling);
        kernel.Process();
    }
}