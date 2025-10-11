/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_HCGEMMBATCHED_TILING_DATA_H
#define ASCEND_OPS_HCGEMMBATCHED_TILING_DATA_H

#include <cstdint>

namespace AsdSip {

struct HCgemmBatchedTilingData {
    int64_t m;
    int64_t k;
    int64_t n;
    int64_t batch;
    int64_t subBatch;
    int64_t miniBatch;
};
}  // namespace AsdSip

#endif  // ASCEND_OPS_HCGEMMBATCHED_TILING_DATA_H