/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "lib/matrix/matmul/matmul.h"
#include "../tiling/interpolation_tiling_data.h"
#include "interpolation_aiv.h"
#include "interpolation_aic.h"

constexpr int32_t OFFSET_SIZE = 4;

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM + sizeof(AsdSip::InterpolationTilingData));

    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdSip::InterpolationTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->tabInterpNum = (*(const __gm__ int32_t *)(p_tilingdata + 0));
    tilingdata->tabQuantNum = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE));
    tilingdata->batch = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE * 2));
    tilingdata->signalLength = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE * 3));
    tilingdata->interpLength = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE * 4));
    tilingdata->numPerBlock = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE * 5));
    tilingdata->linesPerCore = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE * 6));
    tilingdata->tailLines = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE * 7));
    tilingdata->blocksPerLine = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE * 8));
    tilingdata->usedCubeCoreNum = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE * 9));
    tilingdata->workspaceSize = (*(const __gm__ int32_t *)(p_tilingdata + OFFSET_SIZE * 10));
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    int32_t tilingBlockNum = sizeof(AsdSip::InterpolationTilingData) / 32 + 1;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, tilingBlockNum, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingdata->tabInterpNum = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    tilingdata->tabQuantNum = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE));
    tilingdata->batch = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE * 2));
    tilingdata->signalLength = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE * 3));
    tilingdata->interpLength = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE * 4));
    tilingdata->numPerBlock = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE * 5));
    tilingdata->linesPerCore = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE * 6));
    tilingdata->tailLines = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE * 7));
    tilingdata->blocksPerLine = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE * 8));
    tilingdata->usedCubeCoreNum = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE * 9));
    tilingdata->workspaceSize = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + OFFSET_SIZE * 10));
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __global__ __aicore__ void interpolation(GM_ADDR ffts_addr, GM_ADDR x, GM_ADDR tab, GM_ADDR pos,
                                                    GM_ADDR tabIndex, GM_ADDR out, GM_ADDR workspace, GM_ADDR tilingGm)
{
    if (workspace == nullptr) {
        return;
    }
    set_ffts_base_addr((uint64_t)ffts_addr);

    AsdSip::InterpolationTilingData tilingData;
    InitTilingData(tilingGm, &tilingData);
    TCubeTiling cubeTiling;
    CopyTiling(&cubeTiling, tilingGm);

#ifdef __DAV_C220_CUBE__
    Interpolation::InterpolationAIC<float> op_aic;
    TPipe pipe;
    op_aic.mm.Init(&cubeTiling, &pipe);
    op_aic.Init({x, tab, pos, tabIndex, out}, workspace, &tilingData);
    op_aic.Process();
#elif __DAV_C220_VEC__
    Interpolation::InterpolationAIV<float> op;
    op.Init({x, tab, pos, tabIndex, out}, workspace, &tilingData);
    op.Process();
#endif
}