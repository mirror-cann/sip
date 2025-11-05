/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/status/status.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "dft_c2r.h"
#include "dft_c2r_tiling_data.h"
#include "dft_c2r_tiling.h"
 
namespace AsdSip {
using namespace Mki;
constexpr size_t CALC_TWO = 2;
constexpr int THRESHOLDK = 32768;
constexpr static int CUBE_DATA_COUNT_PER_LOOP_SMALL = 128;
constexpr static int CUBE_DATA_COUNT_PER_LOOP_LARGE = 64;

AspbStatus DftC2RTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &param = AnyCast<OpParam::DftC2R>(launchParam.GetParam());
 
    DftC2RTilingData *tilingDataPtr =
        reinterpret_cast<AsdSip::DftC2RTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                 return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
 
    tilingDataPtr->batchSize = 1;
    tilingDataPtr->m = param.batchSize;
    tilingDataPtr->n = param.fftN;
    tilingDataPtr->k = (param.fftN / CALC_TWO + 1) * CALC_TWO;
    tilingDataPtr->transA = 0;
    tilingDataPtr->transB = 0;
    tilingDataPtr->k0 = CUBE_DATA_COUNT_PER_LOOP_SMALL;
    tilingDataPtr->m0 = CUBE_DATA_COUNT_PER_LOOP_SMALL;
    tilingDataPtr->n0 = CUBE_DATA_COUNT_PER_LOOP_SMALL;
    if (tilingDataPtr->k > THRESHOLDK) {
        tilingDataPtr->m0 = CUBE_DATA_COUNT_PER_LOOP_LARGE;
        tilingDataPtr->n0 = CUBE_DATA_COUNT_PER_LOOP_LARGE;
        tilingDataPtr->k0 = CUBE_DATA_COUNT_PER_LOOP_LARGE;
    }
    uint32_t maxCore = 0;
    maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    int64_t defaultWorkspaceSize = 32;
    if (maxCore == 0) {
        maxCore = 1;
    }
    kernelInfo.SetBlockDim(maxCore);
    kernelInfo.GetScratchSizes().push_back(defaultWorkspaceSize);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();
 
    return AsdSip::ErrorType::ACL_SUCCESS;
}
} // namespace AsdSip