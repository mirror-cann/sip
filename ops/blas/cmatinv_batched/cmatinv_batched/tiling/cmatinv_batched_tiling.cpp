
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "cmatinv_batched_tiling_data.h"
#include "cmatinv_batched.h"
#include "cmatinv_batched_tiling.h"

static constexpr uint32_t COMPLEX_ELENUM = 2;
static constexpr uint32_t MAX_CORE_CNT = 40;
static constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

namespace AsdSip {
    
AsdSip::AspbStatus CmatinvBatchedTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();
    CmatinvBatchedTilingData *tilingDataPtr = (CmatinvBatchedTilingData *)(tilingData);
    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    OpParam::CmatinvBatched param = AnyCast<OpParam::CmatinvBatched>(launchParam.GetParam());
    uint32_t dtype =  static_cast<uint32_t>(param.dtype);
    uint32_t n =  static_cast<uint32_t>(param.n);
    uint32_t batchSize =  static_cast<uint32_t>(param.batchSize);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    vecCoreNum = vecCoreNum > MAX_CORE_CNT ? MAX_CORE_CNT : vecCoreNum;
    vecCoreNum = (vecCoreNum == 0) ? 1 : vecCoreNum;

    // init tiling data
    for (uint32_t i = 0; i < MAX_CORE_CNT; i++) {
        tilingDataPtr->startOffset[i] = 0;
        tilingDataPtr->calNum[i] = 0;
    }

    uint32_t matPerCore = batchSize / vecCoreNum;
    uint32_t remainMatNum = batchSize % vecCoreNum;
    if (matPerCore == 0) {
        for (uint32_t i = 0; i < remainMatNum; i++) {
            tilingDataPtr->calNum[i] = 1;
            tilingDataPtr->startOffset[i] = i * n * n * COMPLEX_ELENUM; // complex element num
        }
    } else {
        uint32_t currComputeNum;
        uint32_t currOffset = 0;
        for (uint32_t i = 0; i < vecCoreNum; i++) {
            if (i < remainMatNum) {
                currComputeNum = matPerCore + 1;
            } else {
                currComputeNum = matPerCore;
            }
            tilingDataPtr->calNum[i] = currComputeNum;
            tilingDataPtr->startOffset[i] = currOffset;
            currOffset += currComputeNum * n * n * COMPLEX_ELENUM; // complex element num
        }
    }
    tilingDataPtr->dtype = dtype;
    tilingDataPtr->n = n;

    kernelInfo.SetBlockDim(vecCoreNum);
    kernelInfo.GetScratchSizes() = {WORKSPACE_SIZE};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}