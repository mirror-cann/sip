/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft_c2c_arch35_tiling.h"
#include "../../../../common/include/tiling/fft_all_mix_tiling_data.h"
#include "params/fft_c2c_arch35.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"

namespace AsdSip {
using namespace Mki;

namespace {
bool HasSupportedFactors(int64_t n)
{
    if (n <= 1) {
        return false;
    }
    constexpr int64_t allowedRadices[] = {2, 3, 5, 7, 11, 13, 17, 19};
    for (int64_t radix : allowedRadices) {
        while (n % radix == 0) {
            n /= radix;
        }
    }
    return n == 1;
}
}

AspbStatus FftC2CArch35MixTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        const auto &param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        FftAllMixTilingData *tilingDataPtr =
            reinterpret_cast<AsdSip::FftAllMixTilingData *>(kernelInfo.GetTilingHostAddr());
        ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        ASDSIP_CHECK(param.isMixedRadix, "FftC2CArch35MixTiling requires mixed-radix mode",
                return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);
        ASDSIP_CHECK(HasSupportedFactors(param.fftN), "FftC2CArch35 mixed kernel requires supported factors",
                return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

        tilingDataPtr->batchSize = param.batchSize;
        tilingDataPtr->fftN = param.fftN;
        tilingDataPtr->radixListLen = param.radixListLen;
        tilingDataPtr->isInverse = param.isInverse;
        tilingDataPtr->outer = 1;
        tilingDataPtr->scaleOut = 0;
        tilingDataPtr->transpose = 0;

        int64_t singleBufferSize = param.batchSize * param.fftN * 2 * sizeof(float);
        int64_t workspaceSize = 2 * singleBufferSize;

        tilingDataPtr->workspaceOffsets[0] = 0;
        tilingDataPtr->workspaceOffsets[1] = singleBufferSize;
        tilingDataPtr->workspaceOffsets[2] = 0;
        tilingDataPtr->workspaceOffsets[3] = 0;
        tilingDataPtr->workspaceOffsets[4] = 0;

        uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
        if (maxCore == 0) {
            maxCore = 1;
        }

        kernelInfo.SetBlockDim(maxCore);
        kernelInfo.GetScratchSizes().push_back(workspaceSize);
        ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();
    }

    return AsdSip::ErrorType::ACL_SUCCESS;
}

} // namespace AsdSip
