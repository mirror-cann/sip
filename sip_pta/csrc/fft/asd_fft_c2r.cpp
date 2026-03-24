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

#include <acl/acl.h>
#include <dlfcn.h>
#include <string>
#include <vector>

namespace sip_pta {

at::Tensor asdFftC2R(const at::Tensor& in, bool isForward)
{
    c10_npu::NPUGuard guard(in.device());

    at::Tensor input = in.is_contiguous() ? in : in.contiguous();

    // 仅支持 COMPLEX64 (对应的实数输出为 FLOAT32)
    TORCH_CHECK(input.scalar_type() == at::kComplexFloat,
                "asd_fft_c2r: Input tensor must be ComplexFloat (COMPLEX64).");

    auto selfShape = input.sizes();
    TORCH_CHECK(selfShape.size() >= sip_pta::DIM_2 && selfShape.size() <= sip_pta::DIM_4,
                "asd_fft_c2r: Input tensor must have 2, 3 or 4 dimensions, but got ",
                selfShape.size());

    op_api::FFTParam param;
    param.direction = isForward ? AsdSip::ASCEND_FFT_FORWARD : AsdSip::ASCEND_FFT_INVERSE;
    param.batchSize = selfShape[0];
    param.fftType = AsdSip::ASCEND_FFT_C2R;
    param.dimType = AsdSip::asdFft1dDimType::ASCEND_FFT_HORIZONTAL;

    std::vector<int64_t> out_shape = selfShape.vec();
    int dims = selfShape.size();
    static constexpr int64_t FACTOR = 2;
    if (dims == sip_pta::DIM_2) {
        out_shape[sip_pta::DIM_1] = (selfShape[sip_pta::DIM_1] - 1) * FACTOR;
        param.fftXSize = out_shape[1];
    } else if (dims == sip_pta::DIM_3) {
        param.fftXSize = selfShape[1];
        out_shape[sip_pta::DIM_1] = param.fftXSize;
        out_shape[sip_pta::DIM_2] = (selfShape[sip_pta::DIM_2] - 1) * FACTOR;
        param.fftYSize = out_shape[sip_pta::DIM_2];
    } else if (dims == sip_pta::DIM_4) {
        param.fftXSize = selfShape[sip_pta::DIM_1];
        param.fftYSize = selfShape[sip_pta::DIM_2];
        out_shape[sip_pta::DIM_1] = param.fftXSize;
        out_shape[sip_pta::DIM_2] = param.fftYSize;
        out_shape[sip_pta::DIM_3] = (selfShape[sip_pta::DIM_3] - 1) * FACTOR;
        param.fftZSize = out_shape[sip_pta::DIM_3];
    }

    at::Tensor output = at::empty(out_shape, input.options().dtype(at::kFloat));

    EXEC_FFT_FUNC(AsdSip::asdFftExecC2R, param, input, output);
    return output;
}

// -----------------------------------------------------------------------------------
// 算子注册
// -----------------------------------------------------------------------------------
TORCH_LIBRARY_FRAGMENT(torch_sip, m)
{
    m.def("asd_fft_c2r(Tensor input, bool is_forward) -> Tensor");
}

TORCH_LIBRARY_IMPL(torch_sip, PrivateUse1, m) { m.impl("asd_fft_c2r", &asdFftC2R); }

} // namespace sip_pta