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


at::Tensor CgemvBatchedGolden(asdBlasOperation_t trans, at::Tensor x, at::Tensor A)
{
    at::Tensor golden;
    at::Tensor inA = A;

    if (trans == asdBlasOperation_t::ASDBLAS_OP_C || trans == asdBlasOperation_t::ASDBLAS_OP_T) {
        inA = at::transpose(inA, 2, 1);
        if(trans == asdBlasOperation_t::ASDBLAS_OP_C) {
            inA = at::conj(inA);
        }
    }
    at::Tensor inx = at::unsqueeze(x, 2);
    if (x.dtype() != at::kComplexFloat) {
        inx = inx.to(at::kComplexFloat);
        inA = inA.to(at::kComplexFloat);
    }
    golden = at::bmm(inA, inx);
    if (x.dtype() != at::kComplexFloat) {
        golden = golden.to(x.dtype());
    }
    golden = at::squeeze(golden, 2);

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

TEST(TestOpCgemvBatched, TestCgemvBatchedNormal)
{
    Status status;
    std::complex<float> alpha = std::complex<float>{1.0, 0.0};
    std::complex<float> beta = std::complex<float>{0.0, 0.0};
    int64_t batch = 3;
    int64_t m = 18;
    int64_t n = 30;
    int64_t lda = m;
    int64_t incx = 1;
    int64_t incy = 1;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, m}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[0];
    Tensor x = context.inTensors[1];
    Tensor y = context.inTensors[2];

    aclTensor *aclA = context.aclInTensors[0];
    aclTensor *aclX = context.aclInTensors[1];
    aclTensor *aclY = context.aclInTensors[2];

    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kComplexFloat);
    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);

    at::Tensor golden = CgemvBatchedGolden(trans, atX, atA);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeCgemvBatchedPlan(handle, trans, m);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasCgemvBatched(handle, trans, m, n, alpha,
                            aclA, lda, aclX, incx, beta, aclY, incy, batch);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);

    void *yDeviceAddr = Mki::GetStorageAddr(aclY);
    aclrtMemcpy(y.hostData, batch * m * sizeof(std::complex<float>),
                yDeviceAddr, batch * m * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    asdBlasDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}


TEST(TestOpCgemvBatched, TestCgemvBatchedConjTrans)
{
    Status status;
    std::complex<float> alpha = std::complex<float>{1.0, 0.0};
    std::complex<float> beta = std::complex<float>{0.0, 0.0};
    int64_t batch = 3;
    int64_t m = 24;
    int64_t n = 18;
    int64_t lda = n;
    int64_t incx = 1;
    int64_t incy = 1;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_C;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, m}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, n}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[0];
    Tensor x = context.inTensors[1];
    Tensor y = context.inTensors[2];

    aclTensor *aclA = context.aclInTensors[0];
    aclTensor *aclX = context.aclInTensors[1];
    aclTensor *aclY = context.aclInTensors[2];

    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kComplexFloat);
    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);

    at::Tensor golden = CgemvBatchedGolden(trans, atX, atA);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeCgemvBatchedPlan(handle, trans, m);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasCgemvBatched(handle, trans, m, n, alpha,
                            aclA, lda, aclX, incx, beta, aclY, incy, batch);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);

    void *yDeviceAddr = Mki::GetStorageAddr(aclY);
    aclrtMemcpy(y.hostData, batch * n * sizeof(std::complex<float>),
                yDeviceAddr, batch * n * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    asdBlasDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpHCgemvBatched, TestHCgemvBatchedNormal)
{
    Status status;
    std::complex<op::fp16_t> alpha = std::complex<op::fp16_t>{1.0, 0.0};
    std::complex<op::fp16_t> beta = std::complex<op::fp16_t>{0.0, 0.0};
    int64_t batch = 7;
    int64_t m = 24;
    int64_t n = 32;
    int64_t lda = m;
    int64_t incx = 1;
    int64_t incy = 1;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, m}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[0];
    Tensor x = context.inTensors[1];
    Tensor y = context.inTensors[2];

    aclTensor *aclA = context.aclInTensors[0];
    aclTensor *aclX = context.aclInTensors[1];
    aclTensor *aclY = context.aclInTensors[2];

    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kComplexHalf);
    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexHalf);

    at::Tensor golden = CgemvBatchedGolden(trans, atX, atA);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeHCgemvBatchedPlan(handle, trans, m);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasHCgemvBatched(handle, trans, m, n, alpha,
                            aclA, lda, aclX, incx, beta, aclY, incy, batch);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);

    void *yDeviceAddr = Mki::GetStorageAddr(aclY);
    aclrtMemcpy(y.hostData, batch * m * sizeof(std::complex<op::fp16_t>),
                yDeviceAddr, batch * m * sizeof(std::complex<op::fp16_t>), ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexHalf);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    asdBlasDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}


TEST(TestOpHCgemvBatched, TestHCgemvBatchedConjTrans)
{
    Status status;
    std::complex<op::fp16_t> alpha = std::complex<op::fp16_t>{1.0, 0.0};
    std::complex<op::fp16_t> beta = std::complex<op::fp16_t>{0.0, 0.0};
    int64_t batch = 7;
    int64_t m = 18;
    int64_t n = 24;
    int64_t lda = n;
    int64_t incx = 1;
    int64_t incy = 1;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_C;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, m}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, n}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[0];
    Tensor x = context.inTensors[1];
    Tensor y = context.inTensors[2];

    aclTensor *aclA = context.aclInTensors[0];
    aclTensor *aclX = context.aclInTensors[1];
    aclTensor *aclY = context.aclInTensors[2];

    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kComplexHalf);
    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexHalf);

    at::Tensor golden = CgemvBatchedGolden(trans, atX, atA);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeHCgemvBatchedPlan(handle, trans, m);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasHCgemvBatched(handle, trans, m, n, alpha,
                            aclA, lda, aclX, incx, beta, aclY, incy, batch);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);

    void *yDeviceAddr = Mki::GetStorageAddr(aclY);
    aclrtMemcpy(y.hostData, batch * n * sizeof(std::complex<op::fp16_t>),
                yDeviceAddr, batch * n * sizeof(std::complex<op::fp16_t>), ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexHalf);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    asdBlasDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}