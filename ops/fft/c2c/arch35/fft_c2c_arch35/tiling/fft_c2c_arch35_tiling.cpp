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

constexpr int32_t INDEX_TWO = 2;
constexpr int32_t INDEX_THREE = 3;
constexpr int32_t INDEX_FOUR = 4;

AspbStatus FftC2CArch35Tiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        const auto &param = AnyCast<OpParam::FftC2CArch35>(launchParam.GetParam());
        FftAllMixTilingData *tilingDataPtr =
            reinterpret_cast<AsdSip::FftAllMixTilingData *>(kernelInfo.GetTilingHostAddr());
        ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
                return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

        tilingDataPtr->batchSize = param.batchSize;
        tilingDataPtr->fftN = param.fftN;
        tilingDataPtr->radixListLen = param.radixListLen;
        tilingDataPtr->isInverse = param.isInverse;

        int64_t tempN = param.fftN;
        int64_t tempBatch = param.batchSize;
        int64_t dftOff = 0;
        int64_t twOff = 0;
        
        for (int32_t s = 0; s < param.radixListLen; s++) {
            int32_t radix = 0;
            if (tempN % 2 == 0) radix = 2;
            else if (tempN % 3 == 0) radix = 3;
            else if (tempN % 5 == 0) radix = 5;
            else if (tempN % 7 == 0) radix = 7;
            else if (tempN % 11 == 0) radix = 11;
            else if (tempN % 13 == 0) radix = 13;
            else if (tempN % 17 == 0) radix = 17;
            else if (tempN % 19 == 0) radix = 19;
            
            tilingDataPtr->radix_arr[s] = radix;
            tilingDataPtr->M_arr[s] = tempN / radix;
            tilingDataPtr->currentBatch_arr[s] = tempBatch;
            tilingDataPtr->dft_offset_arr[s] = dftOff;
            tilingDataPtr->tw_offset_arr[s] = twOff;
            
            tempBatch *= radix;
            dftOff += 2 * radix * radix;
            twOff += 2 * radix * tilingDataPtr->M_arr[s];
            tempN = tilingDataPtr->M_arr[s];
        }

        int64_t singleBufferSize = param.batchSize * param.fftN * 2 * sizeof(float);
        int64_t workspaceSize = 2 * singleBufferSize;

        tilingDataPtr->workspaceOffsets[0] = 0;
        tilingDataPtr->workspaceOffsets[1] = singleBufferSize;
        tilingDataPtr->workspaceOffsets[INDEX_TWO] = 0;
        tilingDataPtr->workspaceOffsets[INDEX_THREE] = 0;
        tilingDataPtr->workspaceOffsets[INDEX_FOUR] = 0;

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
