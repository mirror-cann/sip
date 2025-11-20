/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/utils/status/status.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "common_tiling.h"
#include "cal.h"
#include "cal_tiling_data.h"
#include "cal_tiling.h"

namespace AsdSip {

using namespace Mki;

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;

AsdSip::AspbStatus CalTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    uint8_t *tilingData = kernelInfo.GetTilingHostAddr();

    CommonTilingData *tilingDataPtr = reinterpret_cast<CommonTilingData *>(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;
    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    cubeCoreNum = cubeCoreNum > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : cubeCoreNum;
    OpParam::Cal param = AnyCast<OpParam::Cal>(launchParam.GetParam());

    uint32_t useCoreNum = vecCoreNum;
    uint32_t tilingOffset = sizeof(CommonTilingData);
    switch (param.calType) {
        case OpParam::Cal::CalType::CAL_SSCAL:
        case OpParam::Cal::CalType::CAL_CSSCAL:
            useCoreNum = ConfigCommonTilingData(tilingDataPtr, param.n, vecCoreNum);
            break;
        case OpParam::Cal::CalType::CAL_CSCAL:
            tilingDataPtr->n = static_cast<uint32_t>(param.n);
            tilingOffset = sizeof(uint32_t);
            useCoreNum = cubeCoreNum;
            break;
        default:
            useCoreNum = ConfigCommonTilingData(tilingDataPtr, param.n, vecCoreNum);
            break;
    }

    CalTilingData *calTilingDataPtr = reinterpret_cast<CalTilingData *>(tilingData + tilingOffset);
    calTilingDataPtr->alphaReal = param.alphaReal;
    calTilingDataPtr->alphaImag = param.alphaImag;

    if (useCoreNum == 0) {
        useCoreNum = 1;
    }
    kernelInfo.SetBlockDim(useCoreNum);
    kernelInfo.GetScratchSizes() = {16 * 1024 * 1024};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip