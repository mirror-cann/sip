/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include <gtest/gtest.h>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include <cmath>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "base_api.h"
#include "base_inner_api.h"
#include "blas_api.h"
#include "utils/mem_base_inner.h"

using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float EXTENT_OF_ERROR = 0.001;

at::Tensor ConjGolden(at::Tensor input)
{
    at::Tensor golden;
    golden = at::conj(input);

    return golden;
}
} // namespace

TEST(TestOpConj, TestConjCase0)
{
    Status status;
    int64_t m = 4;
    int64_t n = 6;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});

    TensorContext context;

    int deviceId = 0;
    MkiRtStream stream = OpTestInit(deviceId);
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor input = context.inTensors[0];
    Tensor output = context.inTensors[1];

    at::Tensor atInput = at::from_blob(input.hostData, ToIntArrayRef(input.desc.dims), at::kComplexFloat);
    at::Tensor golden = ConjGolden(atInput);

    AspbStatus blasStatus = Conj(input, output, stream);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    MkiRtStreamSynchronize(stream);
    CopyOutTensorToHost(output);

    at::Tensor ret = at::from_blob(output.hostData, ToIntArrayRef(output.desc.dims), at::kComplexFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }

    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}