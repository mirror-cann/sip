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
 * @brief 复数矩阵逐元素相乘算子: C = A * B
 */
at::Tensor asdBlasComplexMatDot(const at::Tensor& mat_x, const at::Tensor& mat_y)
{
    // 基本数据类型校验
    TORCH_CHECK(mat_x.scalar_type() == at::kComplexFloat && mat_y.scalar_type() == at::kComplexFloat,
                "asd_blas_complex_mat_dot: Only ComplexFloat (Complex64) tensors are supported.");

    // 维度匹配校验
    TORCH_CHECK(mat_x.dim() == sip_pta::DIM_2 && mat_y.dim() == sip_pta::DIM_2,
                "asd_blas_complex_mat_dot: Input tensors must be 2-dimensional.");
    TORCH_CHECK(mat_x.sizes() == mat_y.sizes(),
                "asd_blas_complex_mat_dot: Input tensors must have the same shape.");

    c10_npu::NPUGuard guard(mat_x.device());
    // 解析矩阵维度
    int64_t m = mat_x.size(0);
    int64_t n = mat_x.size(1);

    // 准备输出 Tensor，形状与输入保持一致
    at::Tensor result = at::empty_like(mat_x);

    // 执行底层算子
    EXEC_BLAS_FUNC(AsdSip::asdBlasComplexMatDot, AsdSip::asdBlasMakeComplexMatDotPlan, {},
                   m, n, mat_x, mat_y, result);

    return result;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_complex_mat_dot(Tensor mat_x, Tensor mat_y) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m)
{
    m.impl("asd_blas_complex_mat_dot", asdBlasComplexMatDot);
}

} // namespace sip_pta