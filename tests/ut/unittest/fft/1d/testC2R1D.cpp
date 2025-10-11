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

void FFTC2RGolden(at::Tensor &atInTensor, at::Tensor &atOutTensor, uint64_t n_fft, bool forward = false)
{
    atOutTensor = at::_fft_c2r(forward ? atInTensor.conj() : atInTensor, atInTensor.dim() - 1, 0, n_fft);
}

Status FFTC2RCompare(float atol, float rtol, const aclTensorContext &context, uint64_t n_fft, bool forward = false)
{
    const Tensor &inTensor = context.inTensors.at(0);
    const Tensor &outTensor = context.inTensors.at(1);

    at::Tensor atInTensor;
    at::Tensor atOutTensor;

    atInTensor = at::from_blob(inTensor.hostData, ToIntArrayRef(inTensor.desc.dims), at::kComplexFloat);
    atOutTensor = at::from_blob(outTensor.hostData, ToIntArrayRef(outTensor.desc.dims), at::kFloat);

    at::Tensor atOutRefTensor;

    FFTC2RGolden(atInTensor, atOutRefTensor, n_fft, forward);
    at::Tensor errorIdx = at::logical_not(at::isclose(atOutRefTensor.flatten(), atOutTensor.flatten(), ATOL, RTOL));
    if ((errorIdx.sum().item().toDouble() / errorIdx.numel()) > EXTENT_OF_ERROR) {
        return Status::FailStatus(-1, "judge not equal");
    }

    return Status::OkStatus();
}

void run(int64_t batch, int64_t N_doing, int32_t loop = 1)
{
    int deviceId = 0;
    aclrtStream stream;
    OpTestAclInit(deviceId, &stream);

    int64_t signal = N_doing / 2 + 1;

    aclTensorContext context;
    SVector<TensorDesc> tensorDescs;
    tensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, signal}, {}, 0});
    tensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch, N_doing}, {}, 0});
    OpTestMallocInAclTensors(tensorDescs, context);

    asdFftHandle handle;
    asdFftCreate(handle);
    asdFftMakePlan1D(handle, N_doing, asdFftType::ASCEND_FFT_C2R, asdFftDirection::ASCEND_FFT_INVERSE, batch);
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
    Status status = FFTC2RCompare(0.001, 0.001, context, N_doing);
    ASSERT_EQ(status.Ok(), true);
    if (work_size > 0) {
        aclrtFree(workspaceAddr);
    }

    OpTestEnd(deviceId, context, stream);
}

}  // namespace

namespace {

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward0)
{
    run(1, 8);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward1)
{
    run(9, 8);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward2)
{
    run(300, 143);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward3)
{
    run(500, 128);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward4)
{
    run(1000, 512);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward5)
{
    run(2, 2048);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward6)
{
    run(2, 16384);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward7)
{
    run(2, 65536);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward8)
{
    run(2, 2097152);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward9)
{
    run(2, 2025);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward10)
{
    run(2, 3125);
}

TEST(TestOpFFTC2R1D, TestFFTC2R1DBackward11)
{
    run(2, 78125);
}

}  // namespace
