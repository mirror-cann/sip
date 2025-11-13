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
#include "cgemv_batched_tiling_data.h"
#include "cgemv_batched.h"
#include "cgemv_batched_tiling.h"

static constexpr uint32_t MAX_CORE_CNT = 40;
static constexpr uint32_t WORKSPACE_SIZE = 16 * 1024 * 1024;

namespace AsdSip {
using namespace Mki;
AsdSip::AspbStatus CgemvBatchedTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();
    CgemvBatchedTilingData *tilingDataPtr = (CgemvBatchedTilingData *)(tilingData);
    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    OpParam::CgemvBatched param = AnyCast<OpParam::CgemvBatched>(launchParam.GetParam());
    tilingDataPtr->dtype = static_cast<uint32_t>(param.dataType);
    tilingDataPtr->trans =  static_cast<uint32_t>(param.transType);
    tilingDataPtr->m =  static_cast<uint32_t>(param.m);
    tilingDataPtr->n =  static_cast<uint32_t>(param.n);
    tilingDataPtr->maxMatNum =  static_cast<uint32_t>(param.maxMatNum);

    ASDSIP_LOG(INFO) << "dtype:" << param.dataType;
    ASDSIP_LOG(INFO) << "trans:" << param.transType;
    ASDSIP_LOG(INFO) << "m:" << param.m;
    ASDSIP_LOG(INFO) << "n:" << param.n;
    ASDSIP_LOG(INFO) << "maxMatNum:" << param.maxMatNum;

    for (uint32_t i = 0; i < MAX_CORE_CNT; i++) {
        tilingDataPtr->startMatId[i] = 0;
        tilingDataPtr->calMatNum[i] = 0;
    }
    uint32_t baseNum = static_cast<uint32_t>(param.batchCount) / MAX_CORE_CNT;
    uint32_t remainNum = static_cast<uint32_t>(param.batchCount) % MAX_CORE_CNT;

    uint32_t curMatId = 0;
    for (uint32_t i = 0; i < MAX_CORE_CNT; i++) {
        uint32_t curMatNum = 0;
        if (i < remainNum) {
            curMatNum = baseNum + 1;
        } else {
            curMatNum = baseNum;
        }
        tilingDataPtr->startMatId[i] = curMatId;
        tilingDataPtr->calMatNum[i] = curMatNum;
        curMatId += curMatNum;
    }

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    vecCoreNum = vecCoreNum > MAX_CORE_CNT ? MAX_CORE_CNT : vecCoreNum;
    vecCoreNum = (vecCoreNum == 0) ? 1 : vecCoreNum;
    kernelInfo.SetBlockDim(vecCoreNum);
    kernelInfo.GetScratchSizes() = {WORKSPACE_SIZE};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();
    return AsdSip::ErrorType::ACL_SUCCESS;
}
}