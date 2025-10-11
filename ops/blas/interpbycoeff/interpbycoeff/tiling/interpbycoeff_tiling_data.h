/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_INTERPOLATION_TILING_DATA_H
#define ASCEND_OPS_INTERPOLATION_TILING_DATA_H

#include <cstdint>

namespace AsdSip {
struct InterpByCoeffTilingData {
    int32_t batch;
    int32_t rsNum;              // k
    int32_t totalSubcarrier;    // n
    int32_t interpLength;       // m
    int32_t batchPerCore;
    int32_t tailBatch;
    int32_t splitN;
    int32_t splitLength;
    int32_t blocksPerBatch;
    int32_t blockLength;
    int32_t usedCubeCoreNum;
    int32_t workspaceSize;
    int32_t dataType;
    uint32_t reserved[3];
};
}  // namespace AsdSip

#endif  // ASCEND_OPS_INTERPOLATION_TILING_DATA_H