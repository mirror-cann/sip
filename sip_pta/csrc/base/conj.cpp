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

#include "base_api.h"

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>

namespace AsdSip {
using Tensor = Mki::Tensor;
AspbStatus Conj(const Tensor& inTensor, Tensor& outTensor, void* stream, uint8_t* workspace);
} // namespace AsdSip

namespace sip_pta {
at::Tensor conj(const at::Tensor& x)
{
    c10_npu::NPUGuard guard(x.device());

    at::Tensor z = at::empty_like(x);
    auto sip_stream = c10_npu::getCurrentNPUStream().stream(false);

    MkiConvertInput mki_x = MkiConvertInput{x};
    MkiConvertInput mki_z = MkiConvertInput{z};

    uint8_t* workspace_ptr = nullptr;
    EXEC_FUNC(AsdSip::Conj, mki_x, mki_z, sip_stream, workspace_ptr);
    return z;
}

// 2. 算子注册 (Fragment)
TORCH_LIBRARY_FRAGMENT(torch_sip, m) { m.def("conj(Tensor x) -> Tensor"); }

// 3. 硬件后端实现绑定 (NPU 为 PrivateUse1)
TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("conj", &conj); }
} // namespace sip_pta
