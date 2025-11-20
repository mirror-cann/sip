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

#include "utils/assert.h"
#include "log/log.h"
#include "mki/utils/platform/platform_info.h"
#include "cgemm.h"
#include "cgemm_tiling_data.h"
#include "cgemm_tiling.h"

constexpr int64_t MODE_NO_TRANSPOSE = 0;
constexpr int64_t MODE_TRANSPOSE = 1;
constexpr int64_t MODE_CONJ_TRANSPOSE = 2;

struct CgemmLocalParam {
    int64_t m;
    int64_t n;
    int64_t k;
    int64_t gTransA;
    int64_t gTransB;
    int64_t lda;
    int64_t ldb;
    int64_t ldc;
    int64_t ldaPad;
    int64_t ldbPad;
    float alphaReal;
    float alphaImag;
    float betaReal;
    float betaImag;
};

namespace AsdSip {
using namespace Mki;

AspbStatus CgemmTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    CgemmLocalParam localParam;

    void *tilingData = kernelInfo.GetTilingHostAddr();
    CgemmTilingData *tilingDataPtr = (CgemmTilingData *)(tilingData);

    ASDSIP_CHECK(tilingData != nullptr, "tilingDataPtr should not be empty",
              return AsdSip::ErrorType::ACL_ERROR_INVALID_PARAM);

    uint32_t cubeCoreNum = PlatformInfo::Instance().GetCoreNum(CoreType::CORE_TYPE_CUBE);
    if (cubeCoreNum == 0) {
        cubeCoreNum = 1;
    }

    OpParam::Cgemm param = AnyCast<OpParam::Cgemm>(launchParam.GetParam());
    localParam.m = param.m;
    localParam.n = param.n;
    localParam.k = param.k;
    
    localParam.gTransA = param.transA;
    localParam.gTransB = param.transB;

    localParam.lda = param.lda;
    localParam.ldb = param.ldb;
    localParam.ldc = param.ldc;
    localParam.ldaPad = param.ldaPad;
    localParam.ldbPad = param.ldbPad;

    localParam.alphaReal = param.alpha.real();
    localParam.alphaImag = param.alpha.imag();
    localParam.betaReal = param.beta.real();
    localParam.betaImag = param.beta.imag();

    uint64_t defaultWorkspaceSize = 1024;

    uint64_t workspaceSize = defaultWorkspaceSize;

    tilingDataPtr->m = localParam.m;
    tilingDataPtr->n = localParam.n;
    tilingDataPtr->k = localParam.k;
    tilingDataPtr->transA = localParam.gTransA;
    tilingDataPtr->transB = localParam.gTransB;
    tilingDataPtr->lda = localParam.lda;
    tilingDataPtr->ldb = localParam.ldb;
    tilingDataPtr->ldc = localParam.ldc;
    tilingDataPtr->ldaPad = localParam.ldaPad;
    tilingDataPtr->ldbPad = localParam.ldbPad;

    tilingDataPtr->alphaReal = localParam.alphaReal;
    tilingDataPtr->alphaImag = localParam.alphaImag;
    tilingDataPtr->betaReal = localParam.betaReal;
    tilingDataPtr->betaImag = localParam.betaImag;

    kernelInfo.SetBlockDim(cubeCoreNum);  // 16 * 16 matmul, use single cube core.
    kernelInfo.GetScratchSizes().push_back(workspaceSize);
    ASDSIP_LOG(DEBUG) << "KernelInfo:\n" << kernelInfo.ToString();

    return AsdSip::ErrorType::ACL_SUCCESS;
}
}
