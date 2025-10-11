
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
#include "asdsip.h"
#include "filter_api.h"
#include "utils/mem_base_inner.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
#include <iostream>

using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float EXTENT_OF_ERROR = 0.001;

at::Tensor Convolve1D(at::Tensor signal, at::Tensor kernel) {
    size_t n = signal.numel();
    size_t m = kernel.numel();
    size_t resultSize = n + m - 1;
    at::Tensor result = at::zeros({resultSize}, at::kComplexFloat);

    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < m; ++j) {
            result[i + j] += signal[i] * kernel[j];
        }
    }
    size_t startOffset = (m + 1) / 2;
    at::Tensor golden = result.slice(0, startOffset - 1, n + startOffset - 1, 1);
    return golden;
}


at::Tensor Convolve1DBatched(at::Tensor batchSignal, at::Tensor kernel, int64_t batch, int64_t signalLen) {
    at::Tensor batchResult = at::zeros({batch, signalLen}, at::kComplexFloat);
    for (int64_t bid = 0; bid < batch; bid++) {
        batchResult[bid] = Convolve1D(batchSignal[bid], kernel);
    }
    return batchResult;
}


at::Tensor ConvolveGolden(at::Tensor signal, at::Tensor kernel,
                        int64_t batch, int64_t signalLen, int64_t kernelLen) {
    at::Tensor inSignal = signal;
    at::Tensor inKernel = kernel;

    inKernel = inKernel.to(at::kComplexFloat);
    if (signal.dtype() != at::kComplexFloat) {
        inSignal = inSignal.to(at::kComplexFloat);
    }
    at::Tensor golden = Convolve1DBatched(inSignal, inKernel, batch, signalLen);
    if (signal.dtype() != at::kComplexFloat) {
        golden = golden.to(signal.dtype());
    }
    return golden;
}


int AclInit(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    auto ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}
} // namespace

TEST(TestOpConvolve, TestConvolveComplex64)
{
    Status status;
    int64_t signalLen = 128;
    int64_t kernelLen = 8;
    int64_t batch = 2;
    asdConvolveMode_t mode = asdConvolveMode_t::ASD_CONVOLVE_SAME;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, signalLen}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {kernelLen}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, signalLen}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor signal = context.inTensors[0];
    Tensor kernel = context.inTensors[1];
    Tensor output = context.inTensors[2];

    aclTensor *aclSignal = context.aclInTensors[0];
    aclTensor *aclKernel = context.aclInTensors[1];
    aclTensor *aclOutput = context.aclInTensors[2];

    at::Tensor atSignal = at::from_blob(signal.hostData, ToIntArrayRef(signal.desc.dims), at::kComplexFloat);
    at::Tensor atKernel = at::from_blob(kernel.hostData, ToIntArrayRef(kernel.desc.dims), at::kFloat);

    at::Tensor golden = ConvolveGolden(atSignal, atKernel, batch, signalLen, kernelLen);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdConvolveGetWorkspaceSize(signalLen, kernelLen, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);

    AspbStatus opsStatus = asdConvolve(aclSignal, aclKernel, aclOutput, asdConvolveMode_t::ASD_CONVOLVE_SAME, stream, buffer);
    if (opsStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    aclrtSynchronizeStream(stream);

    void *outputDeviceAddr = Mki::GetStorageAddr(aclOutput);
    aclrtMemcpy(output.hostData, signalLen * batch * sizeof(std::complex<float>),
                outputDeviceAddr, signalLen * batch * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(output.hostData, ToIntArrayRef(output.desc.dims), at::kComplexFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}


TEST(TestOpConvolve, TestConvolveComplex32)
{
    Status status;
    int64_t signalLen = 128;
    int64_t kernelLen = 8;
    int64_t batch = 2;
    asdConvolveMode_t mode = asdConvolveMode_t::ASD_CONVOLVE_SAME;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, signalLen}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT16, TENSOR_FORMAT_ND, {kernelLen}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, signalLen}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor signal = context.inTensors[0];
    Tensor kernel = context.inTensors[1];
    Tensor output = context.inTensors[2];

    aclTensor *aclSignal = context.aclInTensors[0];
    aclTensor *aclKernel = context.aclInTensors[1];
    aclTensor *aclOutput = context.aclInTensors[2];

    at::Tensor atSignal = at::from_blob(signal.hostData, ToIntArrayRef(signal.desc.dims), at::kComplexHalf);
    at::Tensor atKernel = at::from_blob(kernel.hostData, ToIntArrayRef(kernel.desc.dims), at::kHalf);

    at::Tensor golden = ConvolveGolden(atSignal, atKernel, batch, signalLen, kernelLen);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdConvolveGetWorkspaceSize(signalLen, kernelLen, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);

    AspbStatus opsStatus = asdConvolve(aclSignal, aclKernel, aclOutput, asdConvolveMode_t::ASD_CONVOLVE_SAME, stream, buffer);
    if (opsStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    aclrtSynchronizeStream(stream);

    void *outputDeviceAddr = Mki::GetStorageAddr(aclOutput);
    aclrtMemcpy(output.hostData, signalLen * batch * sizeof(std::complex<op::fp16_t>),
                outputDeviceAddr, signalLen * batch * sizeof(std::complex<op::fp16_t>), ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(output.hostData, ToIntArrayRef(output.desc.dims), at::kComplexHalf);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}