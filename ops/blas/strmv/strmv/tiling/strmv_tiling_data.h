/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCEND_OPS_STRMV_TILING_DATA_H
#define ASCEND_OPS_STRMV_TILING_DATA_H

#include <cstdint>

namespace AsdSip {

struct StrmvTilingData {
    uint32_t matSizeN;  // 矩阵阶
    uint32_t uplo;      // 上三角矩阵or下三角矩阵
    uint32_t trans;     // 是否转置
    uint32_t diag;      // 是否为单位矩阵
    uint32_t lda;       // 行间隔
    uint32_t incx;      // 行间隔
    uint32_t m0;        // 分块
};
}  // namespace AsdSip

#endif  // ASCEND_OPS_STRMV_TILING_DATA_H