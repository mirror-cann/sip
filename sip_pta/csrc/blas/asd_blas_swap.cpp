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
#include <tuple>

namespace sip_pta {

/**
 * @brief 向量交换算子 (Swap): x <-> y，原地更新两个向量
 */
std::tuple<at::Tensor, at::Tensor> asdBlasSwap(at::Tensor& x, at::Tensor& y)
{
    // 基本校验：检查数据类型是否一致，是否为支持的单精度实数或复数，以及形状是否匹配
    TORCH_CHECK(x.scalar_type() == y.scalar_type(), "asd_blas_swap: Tensors x and y must have the same dtype.");
    TORCH_CHECK(x.sizes() == y.sizes(), "asd_blas_swap: Tensors x and y must have the same shape.");
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kComplexFloat,
                "asd_blas_swap: Tensors must be Float32 or ComplexFloat.");
    c10::DeviceGuard guard(x.device());
    // 参数准备
    int64_t n = x.numel();
    int64_t incx = 1;
    int64_t incy = 1;

    std::vector<int64_t> params = {};

    // 根据数据类型分发底层调用
    if (x.scalar_type() == at::kFloat) {
        EXEC_BLAS_FUNC(AsdSip::asdBlasSswap, AsdSip::asdBlasMakeSwapPlan, params, n, x, incx, y, incy);
    } else {
        EXEC_BLAS_FUNC(AsdSip::asdBlasCswap, AsdSip::asdBlasMakeSwapPlan, params, n, x, incx, y, incy);
    }

    return std::make_tuple(x, y);
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_swap(Tensor(a!) x, Tensor(b!) y) -> (Tensor(a!), Tensor(b!))");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m)
{
    m.impl("asd_blas_swap", &asdBlasSwap);
}

} // namespace sip_pta