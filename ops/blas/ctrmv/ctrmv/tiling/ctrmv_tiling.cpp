
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/utils/status/status.h>
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "ctrmv_tiling_data.h"
#include "ctrmv.h"
#include "ctrmv_tiling.h"

namespace AsdSip {
using namespace Mki;

static constexpr int64_t BASIC_DATA_PROC_CNT = 64;
static constexpr uint32_t ELEMENTS_EACH_COMPLEX64 = 2;

AsdSip::AspbStatus CtrmvTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    CtrmvTilingData *tilingDataPtr = (CtrmvTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);

    OpParam::Ctrmv param = AnyCast<OpParam::Ctrmv>(launchParam.GetParam());

    tilingDataPtr->mode = param.uplo;
    tilingDataPtr->trans = param.trans;
    tilingDataPtr->diag = param.diag;
    tilingDataPtr->n = param.n;
    tilingDataPtr->lda = param.lda;
    tilingDataPtr->incx = param.incx;
    tilingDataPtr->n0 = BASIC_DATA_PROC_CNT;

    uint64_t workspaceSize = param.n * ELEMENTS_EACH_COMPLEX64 * sizeof(float);
    int64_t nDupNum = (param.n - 1) / BASIC_DATA_PROC_CNT + 1;
    int64_t groupDim = nDupNum * nDupNum;

    // 物理核数
    groupDim = groupDim < cubeCoreNum ? groupDim : cubeCoreNum;
    if (groupDim == 0) {
        groupDim = 1;
    }
    kernelInfo.SetBlockDim(groupDim);
    kernelInfo.GetScratchSizes().push_back(workspaceSize);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}
