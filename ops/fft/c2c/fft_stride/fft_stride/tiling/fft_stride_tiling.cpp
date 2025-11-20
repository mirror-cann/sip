/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
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
#include "fft_stride.h"
#include "fft_stride_tiling.h"

namespace AsdSip {
constexpr int32_t S0_NUM128 = 128;
constexpr int32_t S0_NUM32768 = 32768;
constexpr int32_t S0_NUM4096 = 4096;
constexpr int32_t S0_NUM256 = 256;
constexpr int32_t S0_NUM16384 = 16384;
constexpr int32_t S0_NUM8192 = 8192;
constexpr int32_t S0_NUM512 = 512;
constexpr int32_t S0_NUM2048 = 2048;
constexpr int32_t S0_NUM1024 = 1024;
constexpr int32_t CACUL_TWO = 2;
using namespace Mki;
AsdSip::AspbStatus FftStrideTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &param = AnyCast<OpParam::FftStride>(launchParam.GetParam());

    FftStrideTilingData *tilingDataPtr =
        reinterpret_cast<AsdSip::FftStrideTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    tilingDataPtr->batchSize = param.batchSize;
    tilingDataPtr->fftN = param.fftN;
    tilingDataPtr->fftStride = param.strideSize;

    tilingDataPtr->iterCount = param.radixVec.size();
    ASDSIP_CHECK(sizeof(tilingDataPtr->radixVec) / sizeof(tilingDataPtr->radixVec[0]) >= param.radixVec.size(),
            "RadixVec in tilingDataPtr is more smaller in param!", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
    for (uint32_t i = 0; i < param.radixVec.size(); i++) {
        tilingDataPtr->radixVec[i] = param.radixVec[i];
    }

    tilingDataPtr->s0 = S0_NUM128;

    if (param.fftN == S0_NUM32768 && param.strideSize >= S0_NUM4096) {
        tilingDataPtr->s0 = S0_NUM256;
    } else if (param.fftN == S0_NUM16384 && param.strideSize >= S0_NUM8192) {
        tilingDataPtr->s0 = S0_NUM512;
    } else if (param.fftN <= S0_NUM8192 && param.fftN >= S0_NUM2048) {
        if (param.strideSize >= S0_NUM512) {
            tilingDataPtr->s0 = S0_NUM512;
        } else if (param.strideSize >= S0_NUM256) {
            tilingDataPtr->s0 = S0_NUM256;
        } else {
            tilingDataPtr->s0 = S0_NUM128;
        }
    } else {
        if (param.strideSize >= S0_NUM1024) {
            tilingDataPtr->s0 = S0_NUM1024;
        } else if (param.strideSize >= S0_NUM512) {
            tilingDataPtr->s0 = S0_NUM512;
        } else if (param.strideSize >= S0_NUM256) {
            tilingDataPtr->s0 = S0_NUM256;
        } else {
            tilingDataPtr->s0 = S0_NUM128;
        }
    }

    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    if (maxCore == 0) {
        maxCore = 1;
    }
    kernelInfo.SetBlockDim(maxCore);
    kernelInfo.GetScratchSizes().push_back(param.fftN * tilingDataPtr->s0 * CACUL_TWO * sizeof(float) * CACUL_TWO);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip