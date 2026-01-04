/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <complex>

#include <mki/utils/status/status.h>
#include <mki/utils/platform/platform_info.h>
#include <mki/utils/SVector/SVector.h>
#include "utils/assert.h"
#include "log/log.h"
#include "tiling_data.h"
#include "ddd.h"
#include "ddd_sep_tiling.h"

namespace AsdSip {
using namespace Mki;
AsdSip::AspbStatus DddSepTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &param = AnyCast<OpParam::Ddd>(launchParam.GetParam());

    DddSepTilingData *tilingDataPtr = reinterpret_cast<AsdSip::DddSepTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
            return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    tilingDataPtr->batchSize = param.batchSize;
    tilingDataPtr->fftX = param.fftX;
    tilingDataPtr->fftY = param.fftY;
    tilingDataPtr->fftZ = param.fftZ;

    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE));

    // batchSize * fftX * fftY * fftZ * 2
    uint32_t workspaceSize = param.batchSize * param.fftX * param.fftY * param.fftZ * 2 * sizeof(float);

    kernelInfo.SetBlockDim(maxCore);
    kernelInfo.GetScratchSizes().push_back(workspaceSize);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}

}  // namespace AsdSip
