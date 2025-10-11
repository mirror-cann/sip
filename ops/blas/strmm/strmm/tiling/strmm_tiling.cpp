
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/status/status.h>
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "strmm.h"
#include "strmm_tiling_data.h"
#include "strmm_tiling.h"

constexpr int32_t K_FACTOR_1024 = 1024;
namespace AsdSip {
using namespace Mki;

AsdSip::AspbStatus StrmmTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    StrmmTilingData *tilingDataPtr = (StrmmTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNum == 0) {
        cubeCoreNum = 1;
    }

    OpParam::Strmm param = AnyCast<OpParam::Strmm>(launchParam.GetParam());

    uint32_t side = param.side;
    uint32_t uplo = param.uplo;
    uint32_t transa = param.transLeft;
    uint32_t transb = param.transRight;
    uint32_t m = param.m;
    uint32_t n = param.n;
    uint32_t k = param.k;
    uint32_t lessFlag = param.lessFlag;
    float alpha = param.alpha;

    tilingDataPtr->side = side;
    tilingDataPtr->uplo = uplo;
    tilingDataPtr->transa = transa;
    tilingDataPtr->transb = transb;
    tilingDataPtr->m = m;
    tilingDataPtr->n = n;
    tilingDataPtr->k = k;
    tilingDataPtr->lessFlag = lessFlag;
    tilingDataPtr->alpha = alpha;

    kernelInfo.SetBlockDim(cubeCoreNum);  // 16 * 16 matmul, use single cube core.
    kernelInfo.GetScratchSizes().push_back(K_FACTOR_1024);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}