
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <securec.h>
#include <mki/utils/status/status.h>
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "swap.h"
#include "sswap_tiling_data.h"
#include "sswap_tiling.h"

namespace AsdSip {
using namespace Mki;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DEFAULT_VECTOR_CNT = 40;
constexpr uint32_t DEFAULT_CUBE_CNT = 20;

AsdSip::AspbStatus SswapTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    SswapTilingData *tilingDataPtr = (SswapTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNum == 0) {
        cubeCoreNum = 1;
    }
    cubeCoreNum = cubeCoreNum > DEFAULT_CUBE_CNT ? DEFAULT_CUBE_CNT : cubeCoreNum;
    kernelInfo.SetBlockDim(cubeCoreNum);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_CNT ? DEFAULT_VECTOR_CNT : vecCoreNum;

    OpParam::Swap param = AnyCast<OpParam::Swap>(launchParam.GetParam());
    uint32_t n = static_cast<uint32_t>(param.n);
    tilingDataPtr->n = n;
    tilingDataPtr->coreNum = cubeCoreNum;

    uint32_t *startOffset = nullptr;
    try {
        startOffset = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "SswapTiling failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    uint32_t *calNum = nullptr;
    try {
        calNum = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "SswapTiling failed: " << e.what();
        delete[] startOffset;
        startOffset = nullptr;
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    uint32_t numPerCore = n / vecCoreNum;
    uint32_t remainNum = n % vecCoreNum;

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        *(startOffset + i) = 0;
        *(calNum + i) = 0;
    }

    if (numPerCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            *(calNum + i) = 1;
            *(startOffset + i) = i;
        }
    } else {
        uint32_t currOffset = 0;
        uint32_t currCalNum = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currCalNum = numPerCore + 1;
            } else {
                currCalNum = numPerCore;
            }
            *(calNum + i) = currCalNum;
            *(startOffset + i) = currOffset;
            currOffset += currCalNum;
        }
    }
    memcpy_s(tilingDataPtr->startOffset, sizeof(tilingDataPtr->startOffset),
             startOffset, vecCoreNum * sizeof(uint32_t));
    delete[] startOffset;
    startOffset = nullptr;

    memcpy_s(tilingDataPtr->calNum, sizeof(tilingDataPtr->calNum),
             calNum, vecCoreNum * sizeof(uint32_t));
    delete[] calNum;
    calNum = nullptr;
    return AsdSip::ErrorType::ACL_SUCCESS;
}
}