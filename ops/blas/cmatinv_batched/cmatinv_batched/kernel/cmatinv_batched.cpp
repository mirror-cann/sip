/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "cmatinv_batched.h"

extern "C" __global__ __aicore__ void cmatinv_batched(
    GM_ADDR dA, GM_ADDR dUniReal, GM_ADDR dUniImag, GM_ADDR dOffset, GM_ADDR dAinv, GM_ADDR workSpace, GM_ADDR tilingGm)
{
    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGm);
    uint32_t dtype = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf)); // 0-c32; 1-c64

    if (dtype > 0) {
        CmatinvBatched::CmatinvBatchedAIV<float> op;
        op.Init(dA, dUniReal, dUniImag, dOffset, dAinv, workSpace, tilingGm);
        op.Process();
    } else {
        CmatinvBatched::CmatinvBatchedAIV<half> op;
        op.Init(dA, dUniReal, dUniImag, dOffset, dAinv, workSpace, tilingGm);
        op.Process();
    }
}