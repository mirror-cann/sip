
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ssyr_tiling_data.h"
#include "ssyr.h"
#include "ssyr_tiling.h"

namespace AsdSip {

using namespace Mki;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t WORKSPACE_SIZE = 1024;

AsdSip::AspbStatus SsyrTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    SsyrTilingData *tilingDataPtr = (SsyrTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    OpParam::Ssyr param = AnyCast<OpParam::Ssyr>(launchParam.GetParam());

    tilingDataPtr->uplo = param.uplo;
    tilingDataPtr->n = param.n;
    tilingDataPtr->alpha = param.alpha;
    tilingDataPtr->coreNum = vecCoreNum;

    kernelInfo.SetBlockDim(vecCoreNum);

    kernelInfo.GetScratchSizes().push_back(WORKSPACE_SIZE);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}