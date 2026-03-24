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
 * @brief 定义点积模式枚举
 */
enum class DotMode : int64_t {
    SDOT = 0,  // 实数点积
    CDOTU = 1, // 复数点积 (不取共轭)
    CDOTC = 2  // 复数共轭点积 (第一个向量取共轭)
};

/**
 * @brief 向量点积算子适配层
 */
at::Tensor asdBlasDot(const at::Tensor& x, const at::Tensor& y, bool conjugate)
{
    // 校验输入基本属性
    TORCH_CHECK(x.numel() == y.numel(),
                "asd_blas_dot: x and y must have the same number of elements.");
    TORCH_CHECK(x.scalar_type() == y.scalar_type(),
                "asd_blas_dot: x and y must have the same scalar type.");

    c10_npu::NPUGuard guard(x.device());
    DotMode mode;

    // 根据数据类型和 conjugate 标志推导枚举值
    if (x.scalar_type() == at::kFloat) {
        TORCH_CHECK(!conjugate, "asd_blas_dot: conjugate cannot be true for Float32.");
        mode = DotMode::SDOT;
    } else if (x.scalar_type() == at::kComplexFloat) {
        mode = conjugate ? DotMode::CDOTC : DotMode::CDOTU;
    } else {
        TORCH_CHECK(false, "asd_blas_dot: Unsupported type, expected Float32 or Complex64.");
    }

    int64_t n = x.numel();
    const int64_t incx = 1;
    const int64_t incy = 1;
    at::Tensor result = at::empty({1}, x.options());

    std::vector<int64_t> planParam = {};
    // 根据枚举语义分发底层算子
    switch (mode) {
        case DotMode::SDOT:
            EXEC_BLAS_FUNC(AsdSip::asdBlasSdot, asdBlasMakeDotPlan, planParam, n, x, incx, y, incy,
                           result);
            break;
        case DotMode::CDOTU:
            EXEC_BLAS_FUNC(AsdSip::asdBlasCdotu, asdBlasMakeDotPlan, planParam, n, x, incx, y, incy,
                           result);
            break;
        case DotMode::CDOTC:
            EXEC_BLAS_FUNC(AsdSip::asdBlasCdotc, asdBlasMakeDotPlan, planParam, n, x, incx, y, incy,
                           result);
            break;
    }

    return result;
}

// 注册算子
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_dot(Tensor x, Tensor y, bool conjugate=False) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_dot", &asdBlasDot); }
} // namespace sip_pta