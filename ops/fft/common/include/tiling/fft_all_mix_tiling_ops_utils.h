/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDOPS_FFT_ALL_MIX_TILING_OPS_UTILS
#define ASDOPS_FFT_ALL_MIX_TILING_OPS_UTILS

#include "fft_all_mix.h"
#include "fft_all_mix_tiling_data.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "utils/aspb_status.h"

namespace AsdSip {
constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_THREE = 3;
constexpr int32_t INDEX_FOUR = 4;
using namespace Mki;
inline AsdSip::AspbStatus InitFftAllMixTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &param = AnyCast<OpParam::FftAllMix>(launchParam.GetParam());

    FftAllMixTilingData *tilingDataPtr =
        reinterpret_cast<AsdSip::FftAllMixTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    tilingDataPtr->batchSize = param.batchSize;
    tilingDataPtr->fftN = param.fftN;
    tilingDataPtr->radixListLen = param.radixListLen;
    tilingDataPtr->isInverse = param.isInverse;

    tilingDataPtr->workspaceOffsets[0] = 0;
    tilingDataPtr->workspaceOffsets[1] = tilingDataPtr->workspaceOffsets[0] + param.workspace_input_size;
    tilingDataPtr->workspaceOffsets[INDEX_TWO] = tilingDataPtr->workspaceOffsets[1] + param.workspace_output_size;
    tilingDataPtr->workspaceOffsets[INDEX_THREE] =
        tilingDataPtr->workspaceOffsets[INDEX_TWO] + param.workspace_sync_size;
    tilingDataPtr->workspaceOffsets[INDEX_FOUR] =
        tilingDataPtr->workspaceOffsets[INDEX_THREE] + param.workspace_c2c_output_size;

    uint32_t maxCore = 0;
    if (AsdSip::PlatformInfo::Instance().GetPlatformType() == AsdSip::PlatformType::ASCEND_910B) {
        maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
        if (maxCore == 0) {
            maxCore = 1;
        }
    }

    kernelInfo.SetBlockDim(maxCore);
    kernelInfo.GetScratchSizes().push_back(param.workspace_input_size + param.workspace_output_size +
                                           param.workspace_sync_size + param.workspace_c2c_output_size +
                                           param.workspace_auxil_size);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}

}  // namespace AsdSip

#endif
