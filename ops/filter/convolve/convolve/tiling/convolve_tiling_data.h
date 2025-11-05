/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_CONVOLVE_TILING_DATA_H
#define ASCEND_OPS_CONVOLVE_TILING_DATA_H

#include <cstdint>

namespace AsdSip {

struct ConvolveTilingData {
    int64_t signalLen;
    int64_t kernelLen;
    int64_t batchCount;
    int64_t dataType;
};
}  // namespace AsdSip

#endif  // ASCEND_OPS_CONVOLVE_TILING_DATA_H