
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "../tiling/tiling_data.h"
#include "conj.h"

using namespace AscendC;

inline __aicore__ void InitTilingData(const __gm__ uint8_t *pTilingdata, AsdSip::ConjTilingData *tilingdata)
{
#if defined(__CCE_KT_TEST__) || (__CCE_AICORE__ == 220)
    tilingdata->dataNum = (*(const __gm__ uint32_t *)(pTilingdata + 0));
    tilingdata->coreNum = (*(const __gm__ uint32_t *)(pTilingdata + 4));
    tilingdata->len = (*(const __gm__ uint32_t *)(pTilingdata + 8));
    tilingdata->tail = (*(const __gm__ uint32_t *)(pTilingdata + 12));
#else
    __ubuf__ uint8_t *tilingdataInUb = (__ubuf__ uint8_t *)get_imm(0);
    int32_t tilingBlockNum = sizeof(AsdSip::ConjTilingData) / 32 + 1;
    copy_gm_to_ubuf(((__ubuf__ uint8_t *)tilingdataInUb), pTilingdata, 0, 1, tilingBlockNum, 0, 0);
    pipe_barrier(PIPE_ALL);
    tilingdata->dataNum = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdataInUb + 0));
    tilingdata->coreNum = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdataInUb + 4));
    tilingdata->len = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdataInUb + 8));
    tilingdata->tail = (*(__ubuf__ uint32_t *)((__ubuf__ uint8_t *)tilingdataInUb + 12));
    pipe_barrier(PIPE_ALL);
#endif
}

extern "C" __global__ __aicore__ void conj(GM_ADDR x, GM_ADDR y, GM_ADDR workspace, GM_ADDR tiling)
{
    AsdSip::ConjTilingData tilingData;
    InitTilingData(tiling, &tilingData);

    Conj::Conj op;
    op.Init(x, y, &tilingData);
    op.Process();
}