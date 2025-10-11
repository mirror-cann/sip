/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "complex_mat_dot_aiv.h"

using namespace AscendC;

#ifdef __DAV_C220_VEC__
extern "C" __global__ __aicore__ void complex_mat_dot(GM_ADDR matx, GM_ADDR maty, GM_ADDR aug, GM_ADDR result,
                                                      GM_ADDR tiling_gm)
{
    auto core_idx = get_block_idx() * get_subblockdim() + get_subblockid();  // 0 ~ 39
    auto tiling_buf = reinterpret_cast<__gm__ uint8_t *>(tiling_gm);

    uint32_t m = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf));      // num of float elements
    uint32_t n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 4));  // num of float elements

    uint64_t offset = (*(__gm__ uint64_t *)((__gm__ uint8_t *)tiling_buf + 8 + 8 * core_idx));            // FP32
    uint32_t cal_num = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tiling_buf + 8 + 40 * 8 + 4 * core_idx));  // complex num

    ComplexMatDot::ComplexMatDotAIV<float> op;
    op.Init({matx, maty, aug, result, m, n, offset, cal_num});
    op.Process();
}
#endif

#ifdef __DAV_C220_CUBE__
extern "C" __global__ __aicore__ void complex_mat_dot(GM_ADDR matx, GM_ADDR maty, GM_ADDR aug, GM_ADDR result,
                                                      GM_ADDR tiling_gm)
{
}
#endif