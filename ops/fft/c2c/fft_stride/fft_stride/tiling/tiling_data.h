/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_FFTS_TILING_DATA
#define ASDSIP_FFTS_TILING_DATA

constexpr uint32_t MAX_ITERATION = 5;
namespace AsdSip {
struct FftStrideTilingData {
    uint32_t batchSize{0};             // 0
    uint32_t fftN{0};                  // 1
    int32_t fftStride{0};              // 2
    int32_t fftDirection{-1};          // 3
    uint32_t iterCount{0};             // 4
    uint32_t radixVec[MAX_ITERATION];  // 5-9
    uint32_t s0 = 0;                   // 10
};
}
#endif