/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef INTERPBYCOEFF_CONST_H
#define INTERPBYCOEFF_CONST_H

namespace InterpByCoeff {

struct InterpolationKernelParam {
    GM_ADDR x;
    GM_ADDR coeff;
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

#endif  // INTERPBYCOEFF_CONST_H
