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
#include "fft_api.h"
#include "utils/mem_base_inner.h"

using namespace AsdSip;
using namespace Mki;

namespace {

constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float EXTENT_OF_ERROR = 0.001;

void FFTC2R2DGolden(at::Tensor &atInTensor, at::Tensor &atOutTensor, bool forward)
{
    int64_t lastDimSize = (atInTensor.size(atInTensor.dim() - 1) - 1) * 2;
    if (forward) {
        atOutTensor = at::_fft_c2r(atInTensor.conj(), {atInTensor.dim() - 2, atInTensor.dim() - 1}, 0, lastDimSize);
    } else {
        atOutTensor = at::_fft_c2r(atInTensor, {atInTensor.dim() - 2, atInTensor.dim() - 1}, 0, lastDimSize);
    }
}

Status FFTC2R2DCompare(float atol, float rtol, const aclTensorContext &context, bool forward)
{
    const Tensor &inTensor = context.inTensors.at(0);
    const Tensor &outTensor = context.inTensors.at(1);

    at::Tensor atInTensor;
    at::Tensor atOutTensor;

    atInTensor = at::from_blob(inTensor.hostData, ToIntArrayRef(inTensor.desc.dims), at::kComplexFloat);
    atOutTensor = at::from_blob(outTensor.hostData, ToIntArrayRef(outTensor.desc.dims), at::kFloat);

    at::Tensor atOutRefTensor;
    FFTC2R2DGolden(atInTensor, atOutRefTensor, forward);
    at::Tensor errorIdx = at::logical_not(at::isclose(atOutRefTensor.flatten(), atOutTensor.flatten(), ATOL, RTOL));
    if ((errorIdx.sum().item().toDouble() / errorIdx.numel()) > EXTENT_OF_ERROR) {
        return Status::FailStatus(-1, "judge not equal");
    }

    return Status::OkStatus();
}
}  // namespace

namespace {

void run(int64_t batch, int64_t signal_x, int64_t signal_y, int64_t out_signal_y, bool forward)
{
    int deviceId = 0;
    aclrtStream stream;
    OpTestAclInit(deviceId, &stream);

    aclTensorContext context;
    SVector<TensorDesc> tensorDescs;
    tensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, signal_x, signal_y}, {}, 0});
    tensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch, signal_x, out_signal_y}, {}, 0});
    OpTestMallocInAclTensors(tensorDescs, context);

    asdFftHandle handle;
    asdFftCreate(handle);
    asdFftMakePlan2D(handle, signal_x, out_signal_y, asdFftType::ASCEND_FFT_C2R,
                     forward ? asdFftDirection::ASCEND_FFT_FORWARD : asdFftDirection::ASCEND_FFT_INVERSE, batch);
    size_t work_size;
    asdFftGetWorkspaceSize(handle, work_size);
    void *workspaceAddr = nullptr;
    if (work_size > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(work_size), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return);
    }
    asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
    asdFftSetStream(handle, stream);
    asdFftExecC2R(handle, context.aclInTensors[0], context.aclInTensors[1]);
    aclrtSynchronizeStream(stream);
    asdFftDestroy(handle);

    CopyDeviceTensorToHostTensor(context.inTensors[1]);
    Status status = FFTC2R2DCompare(0.001, 0.001, context, forward);
    ASSERT_EQ(status.Ok(), true);
    if (work_size > 0) {
        aclrtFree(workspaceAddr);
    }

    OpTestEnd(deviceId, context, stream);
}

}  // namespace

namespace {

TEST(TestOpFFTC2R2D, TestFFTC2R2DForward)
{
    run(2, 11, 13, (13 - 1) * 2, true);
}

TEST(TestOpFFTC2R2D, TestFFTC2R2DBackward)
{
    run(2, 11, 13, (13 - 1) * 2, false);
}

}  // namespace
