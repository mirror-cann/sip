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
#include "utils/assert.h"
#include "log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "cgerc_tiling_data.h"
#include "cgerc.h"
#include "cgerc_tiling.h"

namespace AsdSip {
using namespace Mki;

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;

AsdSip::AspbStatus CgercTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    CgercTilingData *tilingDataPtr = (CgercTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    uint32_t cubeCoreNumCgerc = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNumCgerc == 0) {
        cubeCoreNumCgerc = 1;
    }
    cubeCoreNumCgerc = cubeCoreNumCgerc > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : cubeCoreNumCgerc;

    OpParam::Cgerc param = AnyCast<OpParam::Cgerc>(launchParam.GetParam());

    uint32_t m = param.m;
    uint32_t n = param.n;
    float alphaReal = param.alphaReal;
    float alphaImag = param.alphaImag;

    uint64_t *startOffset = nullptr;
    try {
        startOffset = new uint64_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "CgercTiling failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    uint64_t *calNum = nullptr;
    try {
        calNum = new uint64_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "CgercTiling failed: " << e.what();
        delete[] startOffset;
        startOffset = nullptr;
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        *(startOffset + i) = 0;
        *(calNum + i) = 0;
    }

    uint64_t rowNumEachCore = m / vecCoreNum;
    uint64_t remainRowNum = m % vecCoreNum;

    // 按复数作tiling，以下单位都是复数
    if (rowNumEachCore == 0) {
        for (uint64_t i = 0; i < remainRowNum; i++) {
            *(calNum + i) = 1;
            *(startOffset + i) = i;
        }
    } else {
        uint64_t currOffset = 0;
        uint64_t currNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainRowNum) {
                currNum = rowNumEachCore + 1;
            } else {
                currNum = rowNumEachCore;
            }
            *(calNum + i) = currNum;
            *(startOffset + i) = currOffset;
            currOffset += currNum;
        }
    }

    tilingDataPtr->m = m;
    tilingDataPtr->n = n;
    tilingDataPtr->alphaReal = alphaReal;
    tilingDataPtr->alphaImag = alphaImag;

    auto ret =
        memcpy_s(tilingDataPtr->startOffset, vecCoreNum * sizeof(uint64_t), startOffset, vecCoreNum * sizeof(uint64_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "startOffset memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ret = memcpy_s(tilingDataPtr->calNum, vecCoreNum * sizeof(uint64_t), calNum, vecCoreNum * sizeof(uint64_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "calNum memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    delete[] startOffset;
    startOffset = nullptr;
    delete[] calNum;
    calNum = nullptr;
    kernelInfo.SetBlockDim(cubeCoreNumCgerc);  // use all aicores
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}