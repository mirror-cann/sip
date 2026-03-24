/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <c10/core/DeviceGuard.h>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "pytorch_npu_helper.hpp"

#include "blas_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>

namespace sip_pta {

/**
 * @brief 实数对称矩阵秩2更新算子 (Ssyr2)
 * 计算公式: A = alpha * x * y^T + alpha * y * x^T + A
 */
at::Tensor asdBlasSsyr2(at::Tensor& A, const at::Tensor& x, const at::Tensor& y,
                        const at::Scalar& alpha, int64_t uplo)
{
    // 形状与维度一致性校验
    TORCH_CHECK(A.dim() == sip_pta::DIM_2 && A.size(0) == A.size(1),
                "asd_blas_ssyr2: Matrix A must be a 2D square tensor.");
    TORCH_CHECK(x.dim() == sip_pta::DIM_1 && y.dim() == sip_pta::DIM_1,
                "asd_blas_ssyr2: Vectors x and y must be 1-dimensional.");
    TORCH_CHECK(A.size(0) == x.size(0) && x.size(0) == y.size(0),
                "asd_blas_ssyr2: Dimension mismatch between A, x, and y.");

    // 数据类型强校验，Ssyr2 算子底层仅支持 Float32
    TORCH_CHECK(A.scalar_type() == at::kFloat && x.scalar_type() == at::kFloat &&
                    y.scalar_type() == at::kFloat,
                "asd_blas_ssyr2: Tensors A, x, and y must be Float32.");

    c10::DeviceGuard guard(A.device());
    TORCH_CHECK(A.is_contiguous() && x.is_contiguous() && y.is_contiguous(),
                "asd_blas_ssyr2: Tensors A, x, and y must be contiguous in memory.");

    int64_t n = A.size(0);
    int64_t lda = A.size(1);
    int64_t incx = 1;
    int64_t incy = 1;

    float alpha_val = alpha.to<float>();
    AsdSip::asdBlasFillMode_t uplo_val = static_cast<AsdSip::asdBlasFillMode_t>(uplo);
    aclTensor* acl_a = CreateAclTensorFromAtTensor(A);
    // 构建空参数集：asdBlasMakeSsyr2Plan 仅需 handle，无需额外参数
    std::vector<int64_t> params = {};

    EXEC_BLAS_FUNC(AsdSip::asdBlasSsyr2, AsdSip::asdBlasMakeSsyr2Plan, params, uplo_val, n,
                   alpha_val, x, incx, y, incy, acl_a, lda);
    return A;
}

// 算子注册模块
// A 发生原地修改，使用 Tensor(a!) 标识其为可变引用
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_ssyr2(Tensor(a!) A, Tensor x, Tensor y, Scalar alpha, int uplo) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_ssyr2", &asdBlasSsyr2); }

} // namespace sip_pta