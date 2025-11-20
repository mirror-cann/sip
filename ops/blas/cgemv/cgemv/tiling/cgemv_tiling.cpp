
/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "mki/utils/platform/platform_info.h"
#include "utils/assert.h"
#include "log/log.h"
#include "cgemv_tiling_data.h"
#include "cgemv.h"
#include "cgemv_tiling.h"

constexpr int32_t K_FACTOR_4 = 4;
constexpr int32_t K_FACTOR_16 = 16;
constexpr int32_t K_FACTOR_1024 = 1024;
namespace AsdSip {
using namespace Mki;
AsdSip::AspbStatus CgemvTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    void *tilingData = kernelInfo.GetTilingHostAddr();

    CgemvTilingData *tilingDataPtr = (CgemvTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNum == 0) {
        cubeCoreNum = 1;
    }

    OpParam::Cgemv param = AnyCast<OpParam::Cgemv>(launchParam.GetParam());
    int64_t transA = param.transA;
    int64_t m = param.M;
    int64_t n = param.N;

    int64_t lda = param.lda;
    int64_t incx = param.incx;
    int64_t incy = param.incy;

    float alphaReal = param.alpha.real();
    float alphaImag = param.alpha.imag();
    float betaReal = param.beta.real();
    float betaImag = param.beta.imag();

    tilingDataPtr->transA = transA;
    tilingDataPtr->m = m;
    tilingDataPtr->n = n;
    tilingDataPtr->alphaReal = alphaReal;
    tilingDataPtr->alphaImag = alphaImag;
    tilingDataPtr->lda = lda;
    tilingDataPtr->incx = incx;
    tilingDataPtr->incy = incy;
    tilingDataPtr->sectionDim = K_FACTOR_4;
    tilingDataPtr->betaReal = betaReal;
    tilingDataPtr->betaImag = betaImag;

    kernelInfo.SetBlockDim(cubeCoreNum);
    kernelInfo.GetScratchSizes() = {K_FACTOR_16 * K_FACTOR_1024 * K_FACTOR_1024};
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}