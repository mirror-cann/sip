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
 * @brief 计算向量绝对值之和 (SASUM)
 * @param x 输入张量
 */
at::Tensor asdBlasAsum(const at::Tensor& x)
{
    c10_npu::NPUGuard guard(x.device());

    at::Tensor x_contiguous = x.is_contiguous() ? x : x.contiguous();
    at::Tensor z = at::empty({1}, x.options().dtype(at::kFloat));

    int64_t n = x_contiguous.numel();
    int64_t incx = 1;

    // 3. 分支逻辑：判断数据类型
    if (x.scalar_type() == at::kComplexFloat) {
        // --- Complex 情况 ---
        EXEC_BLAS_FUNC(AsdSip::asdBlasScasum, // 调用 Scasum
                       AsdSip::asdBlasMakeAsumPlan, {}, n, x_contiguous, incx, z);
    } else if (x.scalar_type() == at::kFloat) {
        // --- Float 情况 ---
        EXEC_BLAS_FUNC(AsdSip::asdBlasSasum, // 调用 Sasum
                       AsdSip::asdBlasMakeAsumPlan, {}, n, x_contiguous, incx, z);
    } else {
        TORCH_CHECK(false, "asd_blas_asum only supports Float32 or Complex64 tensors, but got ",
                    x.scalar_type());
    }

    return z;
}

// 2. 算子注册 (Fragment)
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("asd_blas_asum(Tensor x) -> Tensor"); }
TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_blas_asum", &asdBlasAsum); }
} // namespace sip_pta