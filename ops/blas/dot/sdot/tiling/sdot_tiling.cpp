
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
#include "utils/assert.h"
#include "log/log.h"
#include "sdot_tiling_data.h"
#include "mki/utils/platform/platform_info.h"
#include "dot.h"
#include "sdot_tiling.h"

namespace AsdSip {
using namespace Mki;

constexpr uint32_t BLOCK_SIZE = 32;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;

AsdSip::AspbStatus SdotTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    SdotTilingData *tilingDataPtr = (SdotTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNum == 0) {
        cubeCoreNum = 1;
    }
    cubeCoreNum = cubeCoreNum > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : cubeCoreNum;

    OpParam::Dot param = AnyCast<OpParam::Dot>(launchParam.GetParam());
    uint32_t n = static_cast<uint32_t>(param.n);
    
    uint32_t *startOffset = nullptr;
    try {
        startOffset = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "SdotTiling failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    uint32_t *calNum = nullptr;
    try {
        calNum = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "SdotTiling failed: " << e.what();
        delete[] startOffset;
        startOffset = nullptr;
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        startOffset[i] = 0;
        calNum[i] = 0;
    }

    uint32_t numPerCore = n / vecCoreNum;
    uint32_t remainNum = n % vecCoreNum;

    if (numPerCore != 0) {
        uint32_t currOffset = 0;
        uint32_t currCalNum = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i >= remainNum) {
                currCalNum = numPerCore;
            } else {
                currCalNum = numPerCore + 1;
            }
            calNum[i] = currCalNum;
            startOffset[i] = currOffset;
            currOffset += currCalNum;
        }
    } else {
        for (uint32_t i = 0; i < remainNum; i++) {
            calNum[i] = 1;
            startOffset[i] = i;
        }
    }

    tilingDataPtr->n = n;
    tilingDataPtr->coreNum = vecCoreNum;
    tilingDataPtr->isconj = 0;

    memcpy_s(tilingDataPtr->startOffset, sizeof(tilingDataPtr->startOffset), startOffset,
             vecCoreNum * sizeof(uint32_t));
    memcpy_s(tilingDataPtr->calNum, sizeof(tilingDataPtr->calNum), calNum, vecCoreNum * sizeof(uint32_t));
    delete[] startOffset;
    startOffset = nullptr;
    delete[] calNum;
    calNum = nullptr;
    int64_t defaultWorkspaceSize = 1024;
    kernelInfo.SetBlockDim(cubeCoreNum);
    kernelInfo.GetScratchSizes().push_back(defaultWorkspaceSize);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}