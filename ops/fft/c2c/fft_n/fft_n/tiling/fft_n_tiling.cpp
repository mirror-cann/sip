/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <iostream>

#include <mki/utils/status/status.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"

#include "tiling_data.h"
#include "fft_n.h"
#include "fft_n_tiling.h"

namespace AsdSip {

using Mki::Status;
using Mki::LaunchParam;
using Mki::KernelInfo;
using Mki::AnyCast;
using Mki::PlatformInfo;
using Mki::CoreType;

constexpr int32_t SCRATCH_SIZES = sizeof(float) * 128 * 512 * 20;
constexpr int32_t NFFT_SIZE = 8192;
AsdSip::AspbStatus FftNTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &param = AnyCast<OpParam::FftN>(launchParam.GetParam());

    FftNTilingData *tilingDataPtr = reinterpret_cast<AsdSip::FftNTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    tilingDataPtr->nFFT = param.nFFT;
    tilingDataPtr->batchSize = param.batchSize;
    tilingDataPtr->repeatBatchSize = param.repeatBatchSize;
    tilingDataPtr->iterCount = param.radixVec.size();
    ASDSIP_CHECK(sizeof(tilingDataPtr->radixVec) / sizeof(tilingDataPtr->radixVec[0]) >= param.radixVec.size(),
            "RadixVec in tilingDataPtr is more smaller in param!", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    for (uint32_t i = 0; i < param.radixVec.size(); i++) {
        tilingDataPtr->radixVec[i] = param.radixVec[i];
    }
    // aic aiv mem addr
    for (int64_t i = 0; i < static_cast<int64_t>(param.radixVec.size()); ++i) {
        tilingDataPtr->aicInputAddr[i] = param.aicInputAddr[i];
        tilingDataPtr->aivOutputAddr[i] = param.aivOutputAddr[i];
        tilingDataPtr->lessTCopy[i] = param.lessTCopy[i];
        tilingDataPtr->syncTilingNum[i] = param.syncTilingNum[i];
    }
    tilingDataPtr->fftDirection = (param.isForward != 0) ? static_cast<int32_t>(-1) : static_cast<int32_t>(1);

    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    uint32_t needCoreNum = param.batchSize > maxCore ? maxCore : param.batchSize;
    if (tilingDataPtr->nFFT >= NFFT_SIZE) {
        needCoreNum = maxCore;
    }
    if (needCoreNum == 0) {
        needCoreNum = 1;
    }
    kernelInfo.SetBlockDim(needCoreNum);
    kernelInfo.GetScratchSizes().push_back(SCRATCH_SIZES * needCoreNum);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip