/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <cstdint>
#include <securec.h>
#include "log/log.h"
#include "utils/assert.h"
#include "common_tiling.h"

namespace AsdSip {
constexpr uint64_t BYTENUM_PER_FLOAT32 = 4;
constexpr uint64_t UB_BYTENUM_PER_BLOCK = 32;
constexpr uint64_t ELEMENTS_PER_BLOCK = UB_BYTENUM_PER_BLOCK / BYTENUM_PER_FLOAT32;

// Calculate tiling data value
CommonTilingData CalTilingData(uint32_t totalEleNum, uint32_t vecCoreNum)
{
    CommonTilingData tilingData;
    tilingData.n = totalEleNum;
    tilingData.useCoreNum = 0;

    if (vecCoreNum == 0) {
        ASDSIP_LOG(ERROR) << "vecCoreNum invalid: vecCoreNum = " << vecCoreNum;
        vecCoreNum = 1;
    }

    // Set zero for startOffset and calNum
    for (uint32_t i = 0; i < vecCoreNum; i++) {
        tilingData.startOffset[i] = 0;
        tilingData.calNum[i] = 0;
    }
    // Num of blocks
    uint32_t totalBlockNum = totalEleNum / ELEMENTS_PER_BLOCK;
    // Remain elements num
    uint32_t remainNum = totalEleNum % ELEMENTS_PER_BLOCK;

    if (totalBlockNum == 0) {
        // Use only 1 AIV core.
        tilingData.calNum[0] = remainNum;
        tilingData.useCoreNum = 1;
    } else if (totalBlockNum <= vecCoreNum) {
        for (uint32_t i = 0; i < totalBlockNum; i++) {
            tilingData.startOffset[i] = ELEMENTS_PER_BLOCK * i;
            tilingData.calNum[i] = ELEMENTS_PER_BLOCK;
        }
        tilingData.calNum[totalBlockNum - 1] += remainNum;
        tilingData.useCoreNum = totalBlockNum;
    } else {
        uint64_t blockNumEachCore;
        uint32_t remainBlock;
        
        blockNumEachCore = totalBlockNum / vecCoreNum;
        remainBlock = totalBlockNum % vecCoreNum;

        uint64_t currOffset = 0;
        uint64_t currCalNum = 0;

        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainBlock) {
                currCalNum = (blockNumEachCore + 1) * ELEMENTS_PER_BLOCK;
            } else {
                currCalNum = blockNumEachCore * ELEMENTS_PER_BLOCK;
            }
            tilingData.startOffset[i] = currOffset;
            tilingData.calNum[i] = currCalNum;
            currOffset += currCalNum;
        }
        tilingData.calNum[vecCoreNum - 1] += remainNum;
        tilingData.useCoreNum = vecCoreNum;
    }
    return tilingData;
}

// Set tiling data value
void SetTilingData(CommonTilingData &tilingDataPtr, CommonTilingData tilingData, uint32_t vecCoreNum)
{
    tilingDataPtr.n = tilingData.n;
    tilingDataPtr.useCoreNum = tilingData.useCoreNum;
    auto ret = memcpy_s(tilingDataPtr.startOffset,
        sizeof(tilingDataPtr.startOffset),
        tilingData.startOffset,
        vecCoreNum * sizeof(uint32_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "startOffset memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ret = memcpy_s(tilingDataPtr.calNum, sizeof(tilingDataPtr.calNum),
             tilingData.calNum, vecCoreNum * sizeof(uint32_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "calNum memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
}

uint32_t ConfigCommonTilingData(CommonTilingData *tilingDataPtr, uint32_t totalEleNum, uint32_t vecCoreNum)
{
    CommonTilingData tilingData = CalTilingData(totalEleNum, vecCoreNum);
    SetTilingData(*tilingDataPtr, tilingData, vecCoreNum);
    uint32_t useCoreNum = tilingData.useCoreNum;
    return useCoreNum;
}

}  // namespace AsdSip