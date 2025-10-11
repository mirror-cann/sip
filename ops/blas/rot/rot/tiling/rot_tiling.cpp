/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <securec.h>
#include <mki/utils/platform/platform_info.h>
#include <iostream>
#include "utils/assert.h"
#include "log/log.h"
#include "rot_tiling_data.h"
#include "rot.h"
#include "rot_tiling.h"

namespace AsdSip {

using namespace Mki;
constexpr uint32_t DEFAULT_CUBE_NUM = 20;
constexpr static int64_t ROT_DEFAULT_WORKSPACE_SIZE = 32;

AsdSip::AspbStatus RotTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t coreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    coreNum = coreNum > DEFAULT_CUBE_NUM ? DEFAULT_CUBE_NUM : coreNum;

    OpParam::Rot param = AnyCast<OpParam::Rot>(launchParam.GetParam());

    uint32_t useCoreNum = coreNum;
    if (useCoreNum == 0) {
        useCoreNum = 1;
    }

    RotTilingData *rotTilingDataPtr = (RotTilingData *)(tilingData);
    rotTilingDataPtr->elementCount = param.elementCount;
    rotTilingDataPtr->cosValue = param.cosValue;
    rotTilingDataPtr->sinValue = param.sinValue;

    kernelInfo.SetBlockDim(useCoreNum);
    kernelInfo.GetScratchSizes().push_back(ROT_DEFAULT_WORKSPACE_SIZE);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}  // namespace AsdSip