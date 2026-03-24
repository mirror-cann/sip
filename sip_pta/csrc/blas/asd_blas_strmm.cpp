/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "pytorch_npu_helper.hpp"

#include "blas_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace sip_pta {

/**
 * @brief 三角矩阵-矩阵乘法算子 (Strmm): C = alpha * op(A) * B 或 C = alpha * B * op(A)
 */
at::Tensor asdBlasStrmm(const at::Tensor& A, const at::Tensor& B, const at::Scalar& alpha,
                        int64_t side, int64_t uplo, int64_t trans)
{
    // 基本校验：算子明确要求 FLOAT32
    TORCH_CHECK(A.scalar_type() == at::kFloat && B.scalar_type() == at::kFloat,
                "asd_blas_strmm: Tensors A, B must be of type Float32.");
    c10::DeviceGuard guard(A.device());
    at::Tensor C = at::empty_like(B);
    // 参数准备
    AsdSip::asdBlasSideMode_t sideEnum = static_cast<AsdSip::asdBlasSideMode_t>(side);
    AsdSip::asdBlasFillMode_t uploEnum = static_cast<AsdSip::asdBlasFillMode_t>(uplo);
    AsdSip::asdBlasOperation_t transEnum = static_cast<AsdSip::asdBlasOperation_t>(trans);
    AsdSip::asdBlasDiagType_t diagEnum = AsdSip::asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT;

    int64_t m = B.size(0);
    int64_t n = B.size(1);
    float alpha_f = alpha.to<float>();

    // 根据 side 确定 lda，约束要求：ASDBLAS_SIDE_LEFT 为 m，ASDBLAS_SIDE_RIGHT 为 n
    int64_t lda = (sideEnum == AsdSip::asdBlasSideMode_t::ASDBLAS_SIDE_LEFT) ? m : n;

    // ldb 和 ldc 当前约束要求为 m
    int64_t ldb = m;
    int64_t ldc = m;

    std::vector<int64_t> params = {};

    EXEC_BLAS_FUNC(AsdSip::asdBlasStrmm, asdBlasMakeStrmmPlan, params, sideEnum, uploEnum,
                   transEnum, diagEnum, m, n, alpha_f, A, lda, B, ldb, C, ldc);

    return C;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_strmm(Tensor A, Tensor B, Scalar alpha, int side, int uplo, int trans) -> "
          "Tensor");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_strmm", &asdBlasStrmm); }

} // namespace sip_pta