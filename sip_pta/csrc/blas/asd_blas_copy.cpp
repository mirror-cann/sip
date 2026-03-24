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
 * @brief 向量复制算子核心实现 (带输出参数): out = x
 */
at::Tensor& asdBlasCopyOut(const at::Tensor& x, at::Tensor& out)
{
    // 数据类型校验
    TORCH_CHECK(x.scalar_type() == at::kFloat || x.scalar_type() == at::kComplexFloat,
                "asd_blas_copy: Only Float32 and Complex64 tensors are supported.");
    TORCH_CHECK(x.scalar_type() == out.scalar_type(),
                "asd_blas_copy: Input and output tensors must have the same dtype.");

    // 形状校验
    TORCH_CHECK(x.sizes() == out.sizes(),
                "asd_blas_copy: Input and output tensors must have the same shape.");

    TORCH_CHECK(out.is_contiguous(),
                "asd_blas_copy: The 'out' tensor must be contiguous for in-place writing.");

    c10_npu::NPUGuard guard(x.device());
    at::Tensor x_contig = x.contiguous();

    // 参数准备
    int64_t n = x_contig.numel();
    int64_t incx = 1;
    int64_t incy = 1;

    // 数据分发调用
    if (x_contig.scalar_type() == at::kFloat) {
        EXEC_BLAS_FUNC(AsdSip::asdBlasScopy, AsdSip::asdBlasMakeCopyPlan, {}, n, x_contig, incx, out, incy);
    } else {
        EXEC_BLAS_FUNC(AsdSip::asdBlasCcopy, AsdSip::asdBlasMakeCopyPlan, {}, n, x_contig, incx, out, incy);
    }

    return out;
}

/**
 * @brief 向量复制算子标准实现: y = x
 */
at::Tensor asdBlasCopy(const at::Tensor& x)
{
    at::Tensor y = at::empty_like(x);
    return asdBlasCopyOut(x, y);
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_blas_copy(Tensor x) -> Tensor");
    m.def("asd_blas_copy_out(Tensor x, Tensor(a!) out) -> Tensor(a!)");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m)
{
    m.impl("asd_blas_copy", asdBlasCopy);
    m.impl("asd_blas_copy_out", asdBlasCopyOut);
}

} // namespace sip_pta