/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASCBLAS_KERNEL_UTILS_H
#define ASCBLAS_KERNEL_UTILS_H

#include "ascblas_type.h"

static constexpr int64_t BIT_4 = 4;
static constexpr int64_t BIT_8 = 8;
/**
 * @brief 用于构建AIC和AIV同步的config
 * @param [in] int64_t mode：同步模式
 * @param [in] int64_t flagId：区分不同的同步
 * @return int64_t config：ffts_cross_core_sync第二个参数
 */
__aicore__ inline int64_t GET_FFST_MSG(int64_t mode, int64_t flagId)
{
    return 1 | (mode << BIT_4) | (flagId << BIT_8);
}

/**
 * @brief num 向上按照 paddingNum 的倍数取整
 * @param [in] int64_t num：需要取整的数
 * @param [in] int64_t paddingNum：需要向上取整的最小粒度
 * @return int64_t 向上取整的值
 */
__aicore__ inline int64_t ROUND(int64_t num, int64_t paddingNum)
{
    return ((num + paddingNum - 1) / paddingNum * paddingNum);
}

#include "ascblas_fp32_utils.h"
#endif