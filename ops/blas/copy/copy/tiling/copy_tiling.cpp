/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "copy_tiling.h"
#include <iostream>
#include <mki/utils/status/status.h>
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "copy.h"
#include "common_tiling.h"

constexpr uint32_t DEFAULT_VECTOR_NUM = 40;
constexpr uint64_t WORKSPACE_SIZE = 16777216; // 16 * 1024 * 1024

namespace AsdSip {
using namespace Mki;

AsdSip::AspbStatus CopyTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    CommonTilingData *tilingDataPtr = reinterpret_cast<CommonTilingData *>(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t vecCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_VECTOR);
    vecCoreNum = vecCoreNum > DEFAULT_VECTOR_NUM ? DEFAULT_VECTOR_NUM : vecCoreNum;
    OpParam::Copy param = AnyCast<OpParam::Copy>(launchParam.GetParam());
    uint32_t useCoreNum = ConfigCommonTilingData(tilingDataPtr, param.n, vecCoreNum);
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }

    kernelInfo.SetBlockDim(useCoreNum);
    kernelInfo.GetScratchSizes() = {WORKSPACE_SIZE};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip