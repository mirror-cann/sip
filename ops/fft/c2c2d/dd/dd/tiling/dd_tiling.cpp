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
#include <complex>

#include <mki/utils/status/status.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "tiling_data.h"
#include "dd.h"
#include "dd_tiling.h"

/**
    tiling_args[0] = batchSize;
    tiling_args[1] = fftX;
    tiling_args[2] = fftY;
    tiling_args[3] = batch_nums_per_loop;
*/

constexpr int32_t NUMLOOP_128 = 128;
constexpr int32_t NUMLOOP_64 = 64;
constexpr int32_t NUMLOOP_32 = 32;
constexpr int32_t RET_TWO = 2;
constexpr int32_t RET_ONE = 1;
constexpr int32_t RET_FOUR = 4;
constexpr int32_t RET_EIGHT = 8;
constexpr int32_t RET_FAILED = -1;

namespace {

int32_t CalcBatchNumsPerLoop(int32_t x, int32_t y)
{
    if (x == NUMLOOP_128 && y == NUMLOOP_128) {
        return RET_ONE;
    } else if (x == NUMLOOP_128 && y == NUMLOOP_64) {
        return RET_ONE;
    } else if (x == NUMLOOP_128 && y == NUMLOOP_32) {
        return RET_TWO;
    } else if (x == NUMLOOP_64 && y == NUMLOOP_128) {
        return RET_ONE;
    } else if (x == NUMLOOP_64 && y == NUMLOOP_64) {
        return RET_TWO;
    } else if (x == NUMLOOP_64 && y == NUMLOOP_32) {
        return RET_FOUR;
    } else if (x == NUMLOOP_32 && y == NUMLOOP_128) {
        return RET_TWO;
    } else if (x == NUMLOOP_32 && y == NUMLOOP_64) {
        return RET_FOUR;
    } else if (x == NUMLOOP_32 && y == NUMLOOP_32) {
        return RET_EIGHT;
    } else {
        return RET_FAILED;
    }
}

}

namespace AsdSip {
using namespace Mki;
AsdSip::AspbStatus DdTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &param = AnyCast<OpParam::Dd>(launchParam.GetParam());

    DdTilingData *tilingDataPtr = reinterpret_cast<AsdSip::DdTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
            return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    int64_t batchNumsPerLoop = CalcBatchNumsPerLoop(param.fftX, param.fftY);

    ASDSIP_CHECK(batchNumsPerLoop > 0, "Unsupported fft sizes", return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    tilingDataPtr->batchSize = param.batchSize;
    tilingDataPtr->fftX = param.fftX;
    tilingDataPtr->fftY = param.fftY;
    tilingDataPtr->batchNumsPerLoop = batchNumsPerLoop;

    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));
    uint64_t workspaceSize = param.batchSize * param.fftX * param.fftY * sizeof(std::complex<float>);

    kernelInfo.SetBlockDim(maxCore);
    kernelInfo.GetScratchSizes().push_back(workspaceSize);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}

}  // namespace AsdSip
