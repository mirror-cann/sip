/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_FFTN_TILING_DATA
#define ASDSIP_FFTN_TILING_DATA

constexpr uint32_t MAX_ITERATION = 5;
namespace AsdSip {
struct FftNTilingData {
    uint32_t nFFT{0};                       // 0
    uint32_t batchSize{0};                  // 1
    int32_t repeatBatchSize{0};             // 2
    uint32_t iterCount{0};                  // 3
    uint32_t radixVec[MAX_ITERATION];       // 4-8
    uint32_t aicInputAddr[MAX_ITERATION];   // 9-13
    uint32_t aivOutputAddr[MAX_ITERATION];  // 14-18
    uint32_t lessTCopy[MAX_ITERATION];      // 19-23
    uint32_t syncTilingNum[MAX_ITERATION];  // 24-28
    int32_t fftDirection{-1};               // 29
};
}
#endif