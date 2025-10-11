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

void FFTC2CGolden(at::Tensor &atInTensor, at::Tensor &atOutTensor, bool forward, int32_t strideFlag)
{
    if (strideFlag == 0) {
        atOutTensor = at::_fft_c2c(atInTensor, 0, 0, forward);
    } else {
        atOutTensor = at::_fft_c2c(atInTensor, atInTensor.dim() - 1, 0, forward);
    }
}

Status FFTC2CCompare(float atol, float rtol, const aclTensorContext &context, bool forward = true, int32_t strideFlag = 1)
{
    const Tensor &inTensor = context.inTensors.at(0);
    const Tensor &outTensor = context.inTensors.at(1);

    at::Tensor atInTensor;
    at::Tensor atOutTensor;

    atInTensor = at::from_blob(inTensor.hostData, ToIntArrayRef(inTensor.desc.dims), at::kComplexFloat);
    atOutTensor = at::from_blob(outTensor.hostData, ToIntArrayRef(outTensor.desc.dims), at::kComplexFloat);

    at::Tensor atOutRefTensor;
    FFTC2CGolden(atInTensor, atOutRefTensor, forward, strideFlag);
    at::Tensor isCloseResult = at::isclose(atOutRefTensor, atOutTensor, RTOL, ATOL);
    int numClose = isCloseResult.sum(at::kInt).item<int>();
    int numTotal =  atOutRefTensor.numel();

    if(static_cast<double>(numTotal - numClose) / numTotal > EXTENT_OF_ERROR){
        return Status::FailStatus(-1, "judge not equal");
    }
    return Status::OkStatus();
}
}  // namespace

namespace {
void testC2C1D(int64_t batch, int64_t signal, bool forward = true)
{
    int64_t N_doing = signal;
    int deviceId = 0;
    aclrtStream stream;
    OpTestAclInit(deviceId, &stream);

    aclTensorContext context;
    SVector<TensorDesc> tensorDescs;
    tensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, signal}, {}, 0});
    tensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, signal}, {}, 0});
    OpTestMallocInAclTensors(tensorDescs, context);

    asdFftHandle handle;
    asdFftCreate(handle);
    asdFftMakePlan1D(handle,
        N_doing,
        asdFftType::ASCEND_FFT_C2C,
        forward ? asdFftDirection::ASCEND_FFT_FORWARD : asdFftDirection::ASCEND_FFT_INVERSE,
        batch);

    size_t work_size;
    asdFftGetWorkspaceSize(handle, work_size);
    void *workspaceAddr = nullptr;
    if (work_size > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(work_size), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return);
    }
    asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
    asdFftSetStream(handle, stream);
    asdFftExecC2C(handle, context.aclInTensors[0], context.aclInTensors[1]);
    aclrtSynchronizeStream(stream);

    CopyDeviceTensorToHostTensor(context.inTensors[1]);

    Status status = FFTC2CCompare(0.001, 0.001, context, forward);
    ASSERT_EQ(status.Ok(), true);
    if (work_size > 0) {
        aclrtFree(workspaceAddr);
    }
    asdFftDestroy(handle);
    OpTestEnd(deviceId, context, stream);
}

void testC2C1DLoop(int64_t batch, int64_t signal, bool forward, int32_t loop)
{
    int64_t N_doing = signal;
    int deviceId = 0;
    aclrtStream stream;
    OpTestAclInit(deviceId, &stream);

    aclTensorContext context;
    SVector<TensorDesc> tensorDescs;
    tensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, signal}, {}, 0});
    tensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, signal}, {}, 0});
    OpTestMallocInAclTensors(tensorDescs, context);

    asdFftHandle handle_dummy;
    asdFftCreate(handle_dummy);
    asdFftMakePlan1D(handle_dummy,
        N_doing,
        asdFftType::ASCEND_FFT_C2C,
        forward ? asdFftDirection::ASCEND_FFT_FORWARD : asdFftDirection::ASCEND_FFT_INVERSE,
        batch);

    for (int32_t i = 0; i < loop; i++) {
        asdFftHandle handle;
        asdFftCreate(handle);
        asdFftMakePlan1D(handle,
            N_doing,
            asdFftType::ASCEND_FFT_C2C,
            forward ? asdFftDirection::ASCEND_FFT_FORWARD : asdFftDirection::ASCEND_FFT_INVERSE,
            batch);
            size_t work_size;
            asdFftGetWorkspaceSize(handle, work_size);
            void *workspaceAddr = nullptr;
            if (work_size > 0) {
                auto ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(work_size), ACL_MEM_MALLOC_HUGE_FIRST);
                CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return);
            }
            asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
            asdFftSetStream(handle, stream);
            for (int32_t j = 0; j < loop; j++) {
                asdFftExecC2C(handle, context.aclInTensors[0], context.aclInTensors[1]);
            }
            aclrtSynchronizeStream(stream);
            asdFftDestroy(handle);

            CopyDeviceTensorToHostTensor(context.inTensors[1]);
            Status status = FFTC2CCompare(0.001, 0.001, context, forward);
            ASSERT_EQ(status.Ok(), true);
            if (work_size > 0) {
                aclrtFree(workspaceAddr);
            }
    }
    asdFftDestroy(handle_dummy);
    OpTestEnd(deviceId, context, stream);
}

void testC2C1DStride(int64_t batch, int64_t signal, int64_t N_doing)
{
    int deviceId = 0;
    aclrtStream stream;
    OpTestAclInit(deviceId, &stream);

    aclTensorContext context;
    SVector<TensorDesc> tensorDescs;
    tensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {signal, batch}, {}, 0});
    tensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {signal, batch}, {}, 0});
    OpTestMallocInAclTensors(tensorDescs, context);

    asdFftHandle handle;
    asdFftCreate(handle);
    asdFftMakePlan1D(handle,
        N_doing,
        asdFftType::ASCEND_FFT_C2C,
        asdFftDirection::ASCEND_FFT_FORWARD,
        batch,
        asdFft1dDimType::ASCEND_FFT_VERTICAL);

    size_t work_size;
    asdFftGetWorkspaceSize(handle, work_size);
    void *workspaceAddr = nullptr;
    if (work_size > 0) {
        auto ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(work_size), ACL_MEM_MALLOC_HUGE_FIRST);
        CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("allocate workspace failed. ERROR: %d\n", ret); return);
    }
    asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
    asdFftSetStream(handle, stream);
    asdFftExecC2C(handle, context.aclInTensors[0], context.aclInTensors[1]);
    aclrtSynchronizeStream(stream);

    CopyDeviceTensorToHostTensor(context.inTensors[1]);
    Status status = FFTC2CCompare(0.001, 0.001, context, true, 0);
    ASSERT_EQ(status.Ok(), true);
    if (work_size > 0) {
        aclrtFree(workspaceAddr);
    }
    asdFftDestroy(handle);
    OpTestEnd(deviceId, context, stream);
}
}  // namespace

namespace {

TEST(TestOpFFTC2C1D, TestFFTC2C1DAny)
{
    testC2C1D(2, 258, true);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DUnder256)
{
    testC2C1D(2, 128, true);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DRadix2Medium)
{
    testC2C1D(2, 512, true);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DRadix2Medium8197)
{
    testC2C1D(2, 8197, true);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DRadix2Medium8193)
{
    testC2C1D(2, 8193, true);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DRadix2Medium8192)
{
    testC2C1D(2, 8192, true);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DRadix2Long)
{
    testC2C1D(2, 65536, true);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DMix)
{
    testC2C1D(16, 2178, true);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DAnyLoop)
{
    testC2C1DLoop(2, 258, true, 3);
}

TEST(TestOpFFTC2C1D, TestFFTC21DCUnder256Loop)
{
    testC2C1DLoop(2, 128, true, 3);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DRadix2MediumLoop)
{
    testC2C1DLoop(2, 512, true, 3);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DStride512)
{
    testC2C1DStride(128, 512, 512);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DStride1024)
{
    testC2C1DStride(128, 1024, 1024);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DStride2048)
{
    testC2C1DStride(128, 2048, 2048);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DStride4096)
{
    testC2C1DStride(128, 4096, 4096);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DStride8192)
{
    testC2C1DStride(32, 8192, 8192);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DStride39601)
{
    testC2C1D(32, 199 * 199, true);
}

TEST(TestOpFFTC2C1D, TestFFTC2C1DDft)
{
    testC2C1D(32, 128, true);
}

}  // namespace
