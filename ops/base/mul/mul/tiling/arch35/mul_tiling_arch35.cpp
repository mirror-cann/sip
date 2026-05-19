/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mul_tiling_arch35.h"
#include "mul_tiling_data_arch35.h"
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "mul.h"

namespace AsdSip {
using namespace Mki;

AspbStatus CMulArch35Tiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();
    CMulArch35TilingData *tilingDataPtr = reinterpret_cast<CMulArch35TilingData *>(tilingData);
    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
        return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    OpParam::CMul param = AnyCast<OpParam::CMul>(launchParam.GetParam());
    tilingDataPtr->n = param.n;
    tilingDataPtr->dtype = (param.cMulType == OpParam::CMul::CMulType::MUL_C64) ? DTYPE_C64_ARCH35 : DTYPE_C32_ARCH35;

    int64_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    if (vecCoreNum == 0) {
        vecCoreNum = 1;
    }

    kernelInfo.SetBlockDim(vecCoreNum);
    kernelInfo.GetScratchSizes() = {0};
    ASDSIP_LOG(DEBUG) << "CMulArch35Tiling KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}

}  // namespace AsdSip