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

namespace sip_pta {
/**
 * @brief 复数 Rank-1 更新算子 (CGERC): A = alpha * x * y^H + A
 */
at::Tensor asdBlasCgerc(at::Tensor& A, const at::Tensor& x, const at::Tensor& y,
                        const at::Scalar& alpha)
{
    // 基本校验
    TORCH_CHECK(A.scalar_type() == at::kComplexFloat && x.scalar_type() == at::kComplexFloat &&
                    y.scalar_type() == at::kComplexFloat,
                "asd_blas_cgerc: Only ComplexFloat (Complex64) tensors are supported.");

    c10_npu::NPUGuard guard(A.device());
    // 参数准备
    const int64_t incx = 1;
    const int64_t incy = 1;

    // 解析维度
    // 根据 BLAS 规范，A 是 m x n 矩阵，x 长度为 m，y 长度为 n
    int64_t m = x.size(0);
    int64_t n = y.size(0);
    int64_t lda = A.size(1); // A 为 Row-Major，lda 对应物理列数

    // 转换标量 alpha
    c10::complex<float> a_c10 = alpha.to<c10::complex<float>>();
    std::complex<float> a_std(a_c10.real(), a_c10.imag());

    // 执行算子 (注意参数顺序需匹配底层定义)
    // 目标接口: (handle, m, n, alpha, x, incx, y, incy, A, lda)
    EXEC_BLAS_FUNC(AsdSip::asdBlasCgerc, AsdSip::asdBlasMakeCgercPlan, {}, m, n, a_std, x, incx,
                      y, incy, A, lda);

    return A;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_cgerc(Tensor(a!) A, Tensor x, Tensor y, Scalar alpha) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_cgerc", asdBlasCgerc); }
} // namespace sip_pta