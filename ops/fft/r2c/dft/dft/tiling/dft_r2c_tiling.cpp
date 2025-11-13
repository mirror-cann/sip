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
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "dft_r2c.h"
#include "dft_r2c_tiling_data.h"
#include "dft_r2c_tiling.h"

namespace AsdSip {
using namespace Mki;
constexpr int MUL_DIV_TWO = 2;
constexpr int64_t DFT_R2C_WORKSPACE_SIZE = 32;

AsdSip::AspbStatus DftR2CTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &param = AnyCast<OpParam::DftR2C>(launchParam.GetParam());

    DftR2CTilingData *tilingDataPtr = reinterpret_cast<AsdSip::DftR2CTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    const int thresholdK = 32768;
    const int cubeDataCountPerLoopSmall = 128;
    const int cubeDataCountPerLoopLarge = 64;

    tilingDataPtr->batchSize = 1;
    tilingDataPtr->m = param.batchSize;
    tilingDataPtr->n = (param.fftN / MUL_DIV_TWO + 1) * MUL_DIV_TWO;
    tilingDataPtr->k = param.fftN;
    tilingDataPtr->transA = 0;
    tilingDataPtr->transB = 0;
    tilingDataPtr->m0 = cubeDataCountPerLoopSmall;
    tilingDataPtr->n0 = cubeDataCountPerLoopSmall;
    tilingDataPtr->k0 = cubeDataCountPerLoopSmall;
    if (tilingDataPtr->k > thresholdK) {
        tilingDataPtr->m0 = cubeDataCountPerLoopLarge;
        tilingDataPtr->n0 = cubeDataCountPerLoopLarge;
        tilingDataPtr->k0 = cubeDataCountPerLoopLarge;
    }
    uint32_t maxCore = 0;
    maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    if (maxCore == 0) {
        maxCore = 1;
    }
    kernelInfo.SetBlockDim(maxCore);
    kernelInfo.GetScratchSizes().push_back(DFT_R2C_WORKSPACE_SIZE);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}  // namespace AsdOps