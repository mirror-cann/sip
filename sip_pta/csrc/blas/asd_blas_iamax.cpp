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
 * @brief 向量最大绝对值索引算子适配层 (Isamax, Icamax)
 * 根据输入 Tensor 的数据类型自动派发到底层对应的实数或复数算子
 */
at::Tensor asdBlasIamax(const at::Tensor& x)
{
    // 基础维度校验，约束输入必须为一维张量
    TORCH_CHECK(x.dim() == 1, "asd_blas_iamax: Input tensor x must be 1-dimensional.");

    c10_npu::NPUGuard guard(x.device());
    // 获取元素总数与步长约束
    int64_t n = x.numel();
    const int64_t incx = 1;

    // 创建承载结果的 Tensor
    // 根据算子定义，结果类型强制约束为 INT32，形状为 [1]
    at::Tensor result = at::empty({1}, x.options().dtype(at::kInt));

    std::vector<int64_t> planParam = {};
    // 根据数据类型分发底层算子
    if (x.scalar_type() == at::kFloat) {
        // 单精度实数分支 (Isamax)
        EXEC_BLAS_FUNC(AsdSip::asdBlasIsamax, AsdSip::asdBlasMakeIamaxPlan, planParam, n, x, incx, result);
    } else if (x.scalar_type() == at::kComplexFloat) {
        // 单精度复数分支 (Icamax)
        EXEC_BLAS_FUNC(AsdSip::asdBlasIcamax, AsdSip::asdBlasMakeIamaxPlan, planParam, n, x, incx, result);
    } else {
        // 异常拦截
        TORCH_CHECK(false, "asd_blas_iamax: Unsupported scalar type. Expected Float32 or Complex64.");
    }

    return result;
}

// 算子注册模块
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_iamax(Tensor x) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m)
{
    m.impl("asd_blas_iamax", &asdBlasIamax);
}

} // namespace sip_pta