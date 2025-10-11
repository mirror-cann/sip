
/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <mki/utils/status/status.h>
#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "strmv_tiling_data.h"
#include "strmv.h"
#include "strmv_tiling.h"

namespace AsdSip {
using namespace Mki;

constexpr uint32_t CORE_SPLIT_NUM = 128;

AsdSip::AspbStatus StrmvTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    StrmvTilingData *tilingDataPtr = (StrmvTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);

    const TensorDesc &inTensor = launchParam.GetInTensor(1).desc;
    uint32_t matSizeN = inTensor.dims.at(0);  // 阶
    uint32_t m0 = CORE_SPLIT_NUM;             // 128 分核
    uint32_t m0TileNumOfM = (matSizeN + m0 - 1) / m0;
    uint32_t usedCubeCoreNum = m0TileNumOfM;  // 需要的cube core 个数
    usedCubeCoreNum = usedCubeCoreNum < cubeCoreNum ? usedCubeCoreNum : cubeCoreNum;
    if (usedCubeCoreNum == 0) {
        usedCubeCoreNum = 1;
    }

    OpParam::Strmv param = AnyCast<OpParam::Strmv>(launchParam.GetParam());
    uint32_t lda = param.lda;
    uint32_t incx = param.incx;

    tilingDataPtr->matSizeN = matSizeN;
    tilingDataPtr->uplo = param.uplo;
    tilingDataPtr->trans = param.trans;
    tilingDataPtr->diag = param.diag;
    tilingDataPtr->lda = lda;
    tilingDataPtr->incx = incx;
    tilingDataPtr->m0 = m0;

    int64_t defaultWorkspaceSize = 1024;
    kernelInfo.SetBlockDim(usedCubeCoreNum);  // 16 * 16 matmul
    kernelInfo.GetScratchSizes().push_back(defaultWorkspaceSize);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}
