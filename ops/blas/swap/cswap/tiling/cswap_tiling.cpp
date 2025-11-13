
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
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
#include "cswap_tiling_data.h"
#include "cswap_tiling.h"

namespace AsdSip {
using namespace Mki;
constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t CACUL_TWO = 2;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;

AsdSip::AspbStatus CswapTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();
    CswapTilingData *tilingDataPtr = (CswapTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    uint32_t cubeCoreNumCswap = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNumCswap == 0) {
        cubeCoreNumCswap = 1;
    }
    cubeCoreNumCswap = cubeCoreNumCswap > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : cubeCoreNumCswap;

    OpParam::Swap param = AnyCast<OpParam::Swap>(launchParam.GetParam());

    uint32_t n = static_cast<uint32_t>(param.n);

    uint32_t *startOffset = nullptr;
    try {
        startOffset = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "CswapTiling failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    uint32_t *calNum = nullptr;
    try {
        calNum = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "CswapTiling failed: " << e.what();
        delete[] startOffset;
        startOffset = nullptr;
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        *(startOffset + i) = 0;
        *(calNum + i) = 0;
    }

    uint32_t complexNum = n / 2;
    uint32_t numPerCore = complexNum / vecCoreNum;
    uint32_t remainNum = complexNum % vecCoreNum;

    if (numPerCore == 0) {
        for (uint32_t i = 0; i < remainNum; i++) {
            *(calNum + i) = CACUL_TWO;
            *(startOffset + i) = CACUL_TWO * i;
        }
    } else {
        uint32_t currOffset = 0;
        uint32_t currCalNum = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainNum) {
                currCalNum = (numPerCore + 1) * CACUL_TWO;
            } else {
                currCalNum = numPerCore * CACUL_TWO;
            }
            *(calNum + i) = currCalNum;
            *(startOffset + i) = currOffset;
            currOffset += currCalNum;
        }
    }

    tilingDataPtr->n = n;
    tilingDataPtr->coreNum = vecCoreNum;
    memcpy_s(tilingDataPtr->startOffset, sizeof(tilingDataPtr->startOffset), startOffset,
             vecCoreNum * sizeof(uint32_t));
    memcpy_s(tilingDataPtr->calNum, sizeof(tilingDataPtr->calNum), calNum, vecCoreNum * sizeof(uint32_t));
    delete[] startOffset;
    startOffset = nullptr;
    delete[] calNum;
    calNum = nullptr;
    kernelInfo.SetBlockDim(cubeCoreNumCswap);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return  AsdSip::ErrorType::ACL_SUCCESS;
}
}