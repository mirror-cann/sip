/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "kernel_operator.h"
#include "lib/matrix/matmul/matmul.h"
#include "../tiling/interpbycoeff_tiling_data.h"
#include "interpbycoeff_aiv.h"
#include "interpbycoeff_aic.h"

using namespace AscendC;
using namespace matmul;

__aicore__ inline void CopyTiling(TCubeTiling *tiling, GM_ADDR tilingGM)
{
    uint32_t *ptr = reinterpret_cast<uint32_t *>(tiling);
    auto tiling32 = reinterpret_cast<__gm__ uint32_t *>(tilingGM + sizeof(AsdSip::InterpByCoeffTilingData));

    for (int i = 0; i < sizeof(TCubeTiling) / sizeof(uint32_t); i++, ptr++) {
        *ptr = *(tiling32 + i);
    }
    return;
}

inline __aicore__ void InitTilingData(const __gm__ uint8_t *p_tilingdata, AsdSip::InterpByCoeffTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->batch = (*(const __gm__ int32_t *)(p_tilingdata + 0));
    tilingdata->rsNum = (*(const __gm__ int32_t *)(p_tilingdata + 4));
    tilingdata->totalSubcarrier = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 2));
    tilingdata->interpLength = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 3));
    tilingdata->batchPerCore = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 4));
    tilingdata->tailBatch = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 5));
    tilingdata->splitN = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 6));
    tilingdata->splitLength = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 7));
    tilingdata->blocksPerBatch = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 8));
    tilingdata->blockLength = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 9));
    tilingdata->usedCubeCoreNum = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 10));
    tilingdata->workspaceSize = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 11));
    tilingdata->dataType = (*(const __gm__ int32_t *)(p_tilingdata + 4 * 12));
#else
    __ubuf__ uint8_t *tilingdata_in_ub = (__ubuf__ uint8_t *)get_imm(0);
    int32_t tilingBlockNum = sizeof(AsdSip::InterpByCoeffTilingData) / 32 + 1;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdata_in_ub), p_tilingdata, 0, 1, tilingBlockNum, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingdata->batch = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 0));
    tilingdata->rsNum = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4));
    tilingdata->totalSubcarrier = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 2));
    tilingdata->interpLength = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 3));
    tilingdata->batchPerCore = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 4));
    tilingdata->tailBatch = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 5));
    tilingdata->splitN = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 6));
    tilingdata->splitLength = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 7));
    tilingdata->blocksPerBatch = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 8));
    tilingdata->blockLength = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 9));
    tilingdata->usedCubeCoreNum = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 10));
    tilingdata->workspaceSize = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 11));
    tilingdata->dataType = (*(__ubuf__ int32_t *)((__ubuf__ uint8_t *)tilingdata_in_ub + 4 * 12));
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __global__ __aicore__ void interpbycoeff(GM_ADDR ffts_addr, GM_ADDR x, GM_ADDR coeff, GM_ADDR out,
                                                    GM_ADDR workspace, GM_ADDR tilingGm)
{
    if (workspace == nullptr) {
        return;
    }
    set_ffts_base_addr((uint64_t)ffts_addr);
    AsdSip::InterpByCoeffTilingData tilingData;
    InitTilingData(tilingGm, &tilingData);
    TCubeTiling cubeTiling;
    CopyTiling(&cubeTiling, tilingGm);

    if (tilingData.dataType == 0) {
#ifdef __DAV_C220_CUBE__
        InterpByCoeff::InterpolationAIC<float> op_aic;
        TPipe pipe;
        op_aic.mm.Init(&cubeTiling, &pipe);
        op_aic.Init({x, coeff, out}, workspace, &tilingData);
        op_aic.Process();
#elif __DAV_C220_VEC__
        InterpByCoeff::InterpolationAIV<float> op;
        op.Init({x, coeff, out}, workspace, &tilingData);
        op.Process();
#endif
    } else {
#ifdef __DAV_C220_CUBE__
        InterpByCoeff::InterpolationAIC<half> op_aic;
        TPipe pipe;
        op_aic.mm.Init(&cubeTiling, &pipe);
        op_aic.Init({x, coeff, out}, workspace, &tilingData);
        op_aic.Process();
#elif __DAV_C220_VEC__
        InterpByCoeff::InterpolationAIV<half> op;
        op.Init({x, coeff, out}, workspace, &tilingData);
        op.Process();
#endif
    }
}