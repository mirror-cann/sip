/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "sasum_aiv.h"

constexpr uint8_t CORE_NUM = 40;

using namespace AscendC;

#ifdef __DAV_C220_VEC__
extern "C" __global__ __aicore__ void sasum(GM_ADDR fftsAddr, GM_ADDR inGM, GM_ADDR outGM, GM_ADDR workspace,
                                            GM_ADDR tilingGM)
{
    set_ffts_base_addr((uint64_t)fftsAddr);

    SetSysWorkspace(workspace);
    if (GetSysWorkSpacePtr() == nullptr) {
        return;
    }

    auto vecIdx = get_block_idx() * get_subblockdim() + get_subblockid();

    auto tilingBuf = reinterpret_cast<__gm__ uint8_t *>(tilingGM);

    uint32_t n = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf));
    uint32_t coreNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 4));
    uint32_t offset = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 8 + 4 * vecIdx));
    uint32_t calNum = (*(__gm__ uint32_t *)((__gm__ uint8_t *)tilingBuf + 8 + CORE_NUM * 4 + 4 * vecIdx));

    Sasum::SasumAIV<float> op;

    op.Init(inGM, outGM, n, offset, calNum);
    op.Process();
}

#endif
