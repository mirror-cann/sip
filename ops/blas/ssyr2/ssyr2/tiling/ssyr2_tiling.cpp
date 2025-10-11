
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <securec.h>
#include <cmath>
#include "utils/assert.h"
#include "log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "ssyr2_tiling_data.h"
#include "ssyr2.h"
#include "ssyr2_tiling.h"

constexpr int32_t K_FACTOR_1024 = 1024;
namespace AsdSip {
using namespace Mki;
constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;


AsdSip::AspbStatus Ssyr2Tiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    Ssyr2TilingData *tilingDataPtr = (Ssyr2TilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;

    uint32_t cubeCoreNumSsyr2 = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNumSsyr2 == 0) {
        cubeCoreNumSsyr2 = 1;
    }
    cubeCoreNumSsyr2 = cubeCoreNumSsyr2 > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : cubeCoreNumSsyr2;

    OpParam::Ssyr2 param = AnyCast<OpParam::Ssyr2>(launchParam.GetParam());
    uint32_t uplo = param.uplo;
    uint32_t n = param.n;
    float alpha = param.alpha;

    tilingDataPtr->uplo = uplo;
    tilingDataPtr->n = n;
    tilingDataPtr->alpha = alpha;
    tilingDataPtr->coreNum = vecCoreNum;

    kernelInfo.SetBlockDim(cubeCoreNumSsyr2);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}