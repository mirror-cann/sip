/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDOPS_FFTB_TILING_DATA
#define ASDOPS_FFTB_TILING_DATA

constexpr uint32_t MAX_ITERATION = 10;
namespace AsdSip {
struct FftBTilingData {
    uint32_t nFFT{0};
    uint32_t batchSize{0};
    int32_t fftDirection{-1};
    uint32_t iterCount{0};
    uint32_t radixVec[MAX_ITERATION];
};
}
#endif