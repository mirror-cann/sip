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
#include "caxpy.h"
#include "caxpy_tiling_data.h"
#include "caxpy_tiling.h"

static constexpr uint32_t COMPLEX_NUM = 2;

namespace AsdSip {
using namespace Mki;

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;

AsdSip::AspbStatus CaxpyTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    CaxpyTilingData *tilingDataPtr = (CaxpyTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        ASDSIP_LOG(ERROR) << "Vector core num invalid: get vector core num " << vecCoreNum;
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNum == 0) {
        ASDSIP_LOG(ERROR) << "Cube core num invalid: get cube core num " << cubeCoreNum;
        cubeCoreNum = 1;
    }
    cubeCoreNum = cubeCoreNum > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : cubeCoreNum;

    OpParam::Caxpy param = AnyCast<OpParam::Caxpy>(launchParam.GetParam());

    uint32_t n = param.n;  // 40
    float alphaReal = param.alphaReal;
    float alphaImag = param.alphaImag;

    uint32_t *startOffset = nullptr;
    try {
        startOffset = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "CaxpyTiling failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    uint32_t *calNum = nullptr;
    try {
        calNum = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "CaxpyTiling failed: " << e.what();
        delete[] startOffset;
        startOffset = nullptr;
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        *(startOffset + i) = 0;
        *(calNum + i) = 0;
    }

    uint32_t rowNumEachCore = n / vecCoreNum;  // 2
    uint32_t remainRowNum = n % vecCoreNum;

    if (rowNumEachCore == 0) {
        for (uint32_t i = 0; i < remainRowNum; i++) {
            *(calNum + i) = COMPLEX_NUM;
            *(startOffset + i) = i * COMPLEX_NUM;  // each row has 2*n FP32 elements
        }
    } else {
        uint32_t currOffset = 0;
        uint32_t currNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainRowNum) {
                currNum = rowNumEachCore + 1;
            } else {
                currNum = rowNumEachCore;
            }
            *(calNum + i) = currNum * COMPLEX_NUM;
            *(startOffset + i) = currOffset;
            currOffset += currNum * COMPLEX_NUM;
        }
    }

    tilingDataPtr->n = n * COMPLEX_NUM;  // num of FP32 elements
    tilingDataPtr->alphaReal = alphaReal;
    tilingDataPtr->alphaImag = alphaImag;

    memcpy_s(tilingDataPtr->startOffset, vecCoreNum * sizeof(uint32_t), startOffset, vecCoreNum * sizeof(uint32_t));
    memcpy_s(tilingDataPtr->calNum, vecCoreNum * sizeof(uint32_t), calNum, vecCoreNum * sizeof(uint32_t));
    delete[] startOffset;
    startOffset = nullptr;
    delete[] calNum;
    calNum = nullptr;
    kernelInfo.SetBlockDim(cubeCoreNum);  // use all aicores
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}