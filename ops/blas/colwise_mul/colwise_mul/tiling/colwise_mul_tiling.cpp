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
#include "colwise_mul.h"
#include "colwise_mul_tiling_data.h"
#include "colwise_mul_tiling.h"

namespace AsdSip {
using namespace Mki;

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;

AsdSip::AspbStatus ColwiseMulTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    ColwiseMulTilingData *tilingDataPtr = (ColwiseMulTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNum == 0) {
        cubeCoreNum = 1;
    }
    cubeCoreNum = cubeCoreNum > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : cubeCoreNum;

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    OpParam::ColwiseMul param = AnyCast<OpParam::ColwiseMul>(launchParam.GetParam());
    // matrix m x n in complex64 format
    uint32_t m = param.m;
    uint32_t n = param.n * 2;

    uint32_t *startOffset = nullptr;
    try {
        startOffset = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "ColwiseMulTiling failed: " << e.what();
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    uint32_t *calRowNum = nullptr;
    try {
        calRowNum = new uint32_t[vecCoreNum];
    } catch (std::bad_alloc& e) {
        ASDSIP_LOG(ERROR) << "ColwiseMulTiling failed: " << e.what();
        delete[] startOffset;
        startOffset = nullptr;
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM;
    }

    for (uint32_t i = 0; i < vecCoreNum; i++) {
        *(startOffset + i) = 0;
        *(calRowNum + i) = 0;
    }

    uint32_t rowNumEachCore = m / vecCoreNum;
    uint32_t remainRowNum = m % vecCoreNum;

    if (rowNumEachCore == 0) {
        for (uint32_t i = 0; i < remainRowNum; i++) {
            *(calRowNum + i) = 1;
            *(startOffset + i) = n * i;  // each row has 2*n FP32 elements
        }
    } else {
        uint32_t currOffset = 0;
        uint32_t currRowNum;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainRowNum) {
                currRowNum = rowNumEachCore + 1;
            } else {
                currRowNum = rowNumEachCore;
            }
            *(calRowNum + i) = currRowNum;
            *(startOffset + i) = currOffset;
            currOffset += currRowNum * n;
        }
    }

    tilingDataPtr->m = m;  // num of rows
    tilingDataPtr->n = n;  // num of FP32 elements each row

    auto ret = memcpy_s(
        tilingDataPtr->startOffset, sizeof(tilingDataPtr->startOffset), startOffset, vecCoreNum * sizeof(uint32_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "startOffset memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);
    ret = memcpy_s(tilingDataPtr->calRowNum, sizeof(tilingDataPtr->calRowNum), calRowNum, vecCoreNum * sizeof(uint32_t));
    ASDSIP_CHECK_WITH_NO_RETURN(ret == EOK, "calRowNum memcpy_s failed.", ErrorType::ACL_ERROR_INTERNAL_ERROR);

    delete[] startOffset;
    startOffset = nullptr;
    delete[] calRowNum;
    calRowNum = nullptr;

    kernelInfo.SetBlockDim(cubeCoreNum);  // use all aicores
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}