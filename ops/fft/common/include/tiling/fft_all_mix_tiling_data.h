/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDOPS_FFT_ALL_MIX_TILING_DATA
#define ASDOPS_FFT_ALL_MIX_TILING_DATA

namespace AsdSip {
struct FftAllMixTilingData {
    int64_t batchSize;
    int64_t fftN;
    int32_t radixListLen;
    // for now, c2r is only inverse and inverse = 1, r2c is only forward and inverse = 0
    int32_t isInverse;
    int64_t workspaceOffsets[5];
};
}

#endif