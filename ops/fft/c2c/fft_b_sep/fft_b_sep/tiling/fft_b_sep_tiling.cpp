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
#include "tiling_data.h"
#include "fft_b.h"
#include "fft_b_sep_tiling.h"

namespace AsdSip {

using Mki::Status;
using Mki::LaunchParam;
using Mki::KernelInfo;
using Mki::AnyCast;
using Mki::PlatformInfo;
using Mki::PlatformType;
using Mki::CoreType;

constexpr int32_t SCRATCH_SIZES = sizeof(float) * 655360; // 128 * 256 * 20
AsdSip::AspbStatus FftBSepTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &param = AnyCast<OpParam::FftB>(launchParam.GetParam());

    FftBSepTilingData *tilingDataPtr = reinterpret_cast<AsdSip::FftBSepTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    tilingDataPtr->nFFT = param.nFFT;
    tilingDataPtr->batchSize = param.batchSize;
    tilingDataPtr->fftDirection = (param.isForward != 0) ? static_cast<uint32_t>(0) : static_cast<int32_t>(1);
    tilingDataPtr->iterCount = param.radixVec.size();

    ASDSIP_CHECK(param.radixVec.size() <= sizeof(tilingDataPtr->radixVec) / sizeof(tilingDataPtr->radixVec[0]),
            "RadixVec in tilingDataPtr is more smaller in param!", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    for (uint32_t i = 0; i < param.radixVec.size(); i++) {
        tilingDataPtr->radixVec[i] = param.radixVec[i];
    }
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_910B) {
        uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
        uint32_t needCoreNum = param.batchSize > maxCore ? maxCore : param.batchSize;
        if (needCoreNum == 0) {
            needCoreNum = 1;
        }
        kernelInfo.SetBlockDim(needCoreNum);
        
        size_t workspace = param.nFFT * param.batchSize * sizeof(float) * 4;
        kernelInfo.GetScratchSizes().push_back(workspace);
        ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();
    }
    return AsdSip::ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip