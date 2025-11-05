/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INTERPOLATION_CONST_H
#define INTERPOLATION_CONST_H

namespace Interpolation {

struct InterpolationKernelParam {
    GM_ADDR x;
    GM_ADDR tab;
    GM_ADDR pos;
    GM_ADDR tabIndex;
    GM_ADDR out;
};

constexpr int32_t SYNC_MODE_AIC_AIV = 0;    // inter block synchronization
constexpr int32_t SYNC_MODE_SUB_BLOCK = 1;  // inter subblock synchronization
constexpr int32_t SYNC_MODE_GROUP = 2;      // intra block synchronization

constexpr int32_t AICFLAGID = 0;
constexpr int32_t AIVFLAGID = 1;
constexpr int32_t AIC2AIVFLAGID = 2;
constexpr int32_t AIV2AICFLAGID = 3;
constexpr int32_t AIC_FINISH_FLAG_ID = 6;
constexpr int32_t AIV_FINISH_FLAG_ID = 7;

constexpr int32_t MAX_COUNT = 4096;
}

#endif  // INTERPOLATION_CONST_H
