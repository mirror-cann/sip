/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "swap_last_2axes.h"
#include "swap_last_2axes_tiling_data.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "swap_last_2axes_tiling.h"

namespace AsdSip {
using namespace Mki;
AsdSip::AspbStatus SwapLast2AxesTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    const auto &param = AnyCast<OpParam::SwapLast2Axes>(launchParam.GetParam());

    SwapLast2AxesTilingData *tilingDataPtr =
        reinterpret_cast<AsdSip::SwapLast2AxesTilingData *>(kernelInfo.GetTilingHostAddr());
    ASDSIP_CHECK(tilingDataPtr != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    tilingDataPtr->batch = param.batch;
    tilingDataPtr->row = param.row;
    tilingDataPtr->col = param.col;

    uint32_t maxCore = static_cast<uint32_t>(PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR));
    if (maxCore == 0) {
        maxCore = 1;
    }
    kernelInfo.SetBlockDim(maxCore);
    kernelInfo.GetScratchSizes().push_back(0);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip