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
#include <c10/core/DeviceGuard.h>
#include <torch_npu/csrc/framework/OpCommand.h>
#include "pytorch_npu_helper.hpp"

#include "blas_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>
#include <tuple>

namespace sip_pta {

/**
 * @brief 复数向量旋转算子适配层 (Csrot)
 * 对输入复向量组 (x, y) 进行原地旋转计算
 */
std::tuple<at::Tensor, at::Tensor> asdBlasCsrot(at::Tensor& x, at::Tensor& y, double c, double s)
{
    TORCH_CHECK(x.dim() == 1 && y.dim() == 1, "asd_blas_csrot: Tensors x and y must be 1-dimensional.");
    TORCH_CHECK(x.numel() == y.numel(), "asd_blas_csrot: Tensors x and y must have the same size.");
    TORCH_CHECK(x.scalar_type() == at::kComplexFloat && y.scalar_type() == at::kComplexFloat,
                "asd_blas_csrot: Tensors x and y must be ComplexFloat (Complex64).");

    c10::DeviceGuard guard(x.device());

    TORCH_CHECK(x.is_contiguous() && y.is_contiguous(),
                "asd_blas_csrot: Tensors x and y must be contiguous in memory for in-place updates.");

    // 参数提取
    int64_t n = x.numel();
    const int64_t incx = 1;
    const int64_t incy = 1;

    float c_val = static_cast<float>(c);
    float s_val = static_cast<float>(s);

    std::vector<int64_t> params = {};

    EXEC_BLAS_FUNC(AsdSip::asdBlasCsrot, AsdSip::asdBlasMakeRotPlan, params,
                   n, x, incx, y, incy, c_val, s_val);

    return std::make_tuple(x, y);
}

// 算子注册模块
// 使用 Tensor(a!) 和 Tensor(b!) 标记 x 和 y 为可变引用 (原地更新)
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_csrot(Tensor(a!) x, Tensor(b!) y, float c, float s) -> (Tensor(a!), Tensor(b!))");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m)
{
    m.impl("asd_blas_csrot", &asdBlasCsrot);
}

} // namespace sip_pta