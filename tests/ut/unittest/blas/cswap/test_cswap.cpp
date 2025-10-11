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
#include "blas_api.h"
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

int AclInit(int32_t deviceId, aclrtStream *stream)
{
    // 固定写法，acl初始化
    // auto ret = aclInit(nullptr);
    // CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclInit failed. ERROR: %d\n", ret); return ret);
    auto ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

}  // namespace

TEST(TestOpCswap, TestCswapF32)
{
    int64_t n = 80;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor x = context.inTensors[0];
    Tensor y = context.inTensors[1];

    aclTensor *aclX = context.aclInTensors[0];
    aclTensor *aclY = context.aclInTensors[1];

    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor atY = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    void * goldenX = malloc(n * sizeof(std::complex<float>));
    void * goldenY = malloc(n * sizeof(std::complex<float>));

    memcpy_s(goldenX, n * sizeof(std::complex<float>), y.hostData, n * sizeof(std::complex<float>));
    memcpy_s(goldenY, n * sizeof(std::complex<float>), x.hostData, n * sizeof(std::complex<float>));

    at::Tensor atGoldenX = at::from_blob(goldenX, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor atGoldenY = at::from_blob(goldenY, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeSwapPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasCswap(handle, n, aclX, incx, aclY, incy);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);
    // CopyDeviceTensorToHostTensor(x);
    void *inputXDeviceAddr = Mki::GetStorageAddr(aclX);
    aclrtMemcpy(x.hostData,
        n * sizeof(std::complex<float>),
        inputXDeviceAddr,
        n * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    // CopyDeviceTensorToHostTensor(y);
    void *inputYDeviceAddr = Mki::GetStorageAddr(aclY);
    aclrtMemcpy(y.hostData,
        n * sizeof(std::complex<float>),
        inputYDeviceAddr,
        n * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor retX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor retY = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    if (!at::allclose(atGoldenX, retX, EXTENT_OF_ERROR, EXTENT_OF_ERROR)
                     || !at::allclose(atGoldenY, retY, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    free(goldenX);
    free(goldenY);
    asdBlasDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCswap, TestCswapF32Unaligned)
{
    int64_t n = 163;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor x = context.inTensors[0];
    Tensor y = context.inTensors[1];

    aclTensor *aclX = context.aclInTensors[0];
    aclTensor *aclY = context.aclInTensors[1];

    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor atY = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    void * goldenX = malloc(n * sizeof(std::complex<float>));
    void * goldenY = malloc(n * sizeof(std::complex<float>));

    memcpy_s(goldenX, n * sizeof(std::complex<float>), y.hostData, n * sizeof(std::complex<float>));
    memcpy_s(goldenY, n * sizeof(std::complex<float>), x.hostData, n * sizeof(std::complex<float>));

    at::Tensor atGoldenX = at::from_blob(goldenX, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor atGoldenY = at::from_blob(goldenY, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeSwapPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasCswap(handle, n, aclX, incx, aclY, incy);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);
    asdBlasSynchronize(handle);
    // CopyDeviceTensorToHostTensor(x);
    void *inputXDeviceAddr = Mki::GetStorageAddr(aclX);
    aclrtMemcpy(x.hostData,
        n * sizeof(std::complex<float>),
        inputXDeviceAddr,
        n * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);
    // CopyDeviceTensorToHostTensor(y);
    void *inputYDeviceAddr = Mki::GetStorageAddr(aclY);
    aclrtMemcpy(y.hostData,
        n * sizeof(std::complex<float>),
        inputYDeviceAddr,
        n * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor retX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor retY = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    if (!at::allclose(atGoldenX, retX, EXTENT_OF_ERROR, EXTENT_OF_ERROR)
                     || !at::allclose(atGoldenY, retY, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    free(goldenX);
    free(goldenY);
    asdBlasDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}