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

at::Tensor StrmmGolden(asdBlasSideMode_t side, asdBlasOperation_t trans, float alpha, at::Tensor A, at::Tensor B)
{
    at::Tensor golden;
    if (trans == asdBlasOperation_t::ASDBLAS_OP_T) {
        A = at::transpose(A, 1, 0);
    }

    if (side == asdBlasSideMode_t::ASDBLAS_SIDE_LEFT) {
        golden = at::matmul(A, B);
    } else {
        golden = at::matmul(B, A);
    }

    golden = at::mul(golden, alpha);

    return golden;
}

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

} // namespace


TEST(TestOpStrmm, TestStrmmNoTransAndLeft)
{
    Status status;
    float alpha = 1;
    int64_t m = 3;
    int64_t n = 3;
    int64_t lda = 3;
    int64_t ldb = 3;
    int64_t ldc = 3;
    asdBlasSideMode_t side = asdBlasSideMode_t::ASDBLAS_SIDE_LEFT;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasDiagType_t diag = asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT;

    SVector<TensorDesc> inTensorDescs;     
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, m}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[0];
    Tensor B = context.inTensors[1];
    Tensor C = context.inTensors[2];
    auto dimC = C.desc.dims;

    aclTensor *aclA = context.aclInTensors[0];
    aclTensor *aclB = context.aclInTensors[1];
    aclTensor *aclC = context.aclInTensors[2];

    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kFloat);
    at::Tensor atB = at::from_blob(B.hostData, ToIntArrayRef(B.desc.dims), at::kFloat);
    
    at::Tensor golden = StrmmGolden(side, trans, alpha, atA, atB);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeStrmmPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus =
        asdBlasStrmm(handle, side, uplo, trans, diag, m, n, alpha, aclA, lda, aclB, ldb, aclC, ldc);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);
    asdBlasSynchronize(handle);
    // CopyDeviceTensorToHostTensor(C);
    void *outputCDeviceAddr = Mki::GetStorageAddr(aclC);
    aclrtMemcpy(C.hostData, m * n * sizeof(float), outputCDeviceAddr, m * n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    asdBlasDestroy(handle);

    at::Tensor ret = at::from_blob(C.hostData, ToIntArrayRef(dimC), at::kFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }

    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpStrmm, TestStrmmNoTransAndRight)
{
    Status status;
    float alpha = 2.3456;
    int64_t m = 3;
    int64_t n = 3;
    int64_t lda = 3;
    int64_t ldb = 3;
    int64_t ldc = 3;
    asdBlasSideMode_t side = AsdSip::asdBlasSideMode_t::ASDBLAS_SIDE_RIGHT;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasDiagType_t diag = asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT;

    SVector<TensorDesc> inTensorDescs;     
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[1];
    Tensor B = context.inTensors[0];
    Tensor C = context.inTensors[2];
    auto dimC = C.desc.dims;

    aclTensor *aclA = context.aclInTensors[1];
    aclTensor *aclB = context.aclInTensors[0];
    aclTensor *aclC = context.aclInTensors[2];

    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kFloat);
    at::Tensor atB = at::from_blob(B.hostData, ToIntArrayRef(B.desc.dims), at::kFloat);
    
    at::Tensor golden = StrmmGolden(side, trans, alpha, atA, atB);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeStrmmPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus =
        asdBlasStrmm(handle, side, uplo, trans, diag, m, n, alpha, aclA, lda, aclB, ldb, aclC, ldc);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);
    // CopyDeviceTensorToHostTensor(C);
    void *outputCDeviceAddr = Mki::GetStorageAddr(aclC);
    aclrtMemcpy(C.hostData, m * n * sizeof(float), outputCDeviceAddr, m * n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    asdBlasDestroy(handle);

    at::Tensor ret = at::from_blob(C.hostData, ToIntArrayRef(dimC), at::kFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }

    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpStrmm, TestStrmmTransAndRight)
{
    Status status;
    float alpha = 2.3456;
    int64_t m = 3;
    int64_t n = 3;
    int64_t lda = 3;
    int64_t ldb = 3;
    int64_t ldc = 3;
    asdBlasSideMode_t side = AsdSip::asdBlasSideMode_t::ASDBLAS_SIDE_RIGHT;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_T;
    asdBlasDiagType_t diag = asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT;

    SVector<TensorDesc> inTensorDescs;     
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[1];
    Tensor B = context.inTensors[0];
    Tensor C = context.inTensors[2];
    auto dimC = C.desc.dims;

    aclTensor *aclA = context.aclInTensors[1];
    aclTensor *aclB = context.aclInTensors[0];
    aclTensor *aclC = context.aclInTensors[2];

    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kFloat);
    at::Tensor atB = at::from_blob(B.hostData, ToIntArrayRef(B.desc.dims), at::kFloat);
    
    at::Tensor golden = StrmmGolden(side, trans, alpha, atA, atB);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeStrmmPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus =
        asdBlasStrmm(handle, side, uplo, trans, diag, m, n, alpha, aclA, lda, aclB, ldb, aclC, ldc);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);
    // CopyDeviceTensorToHostTensor(C);
    void *outputCDeviceAddr = Mki::GetStorageAddr(aclC);
    aclrtMemcpy(C.hostData, m * n * sizeof(float), outputCDeviceAddr, m * n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    asdBlasDestroy(handle);

    at::Tensor ret = at::from_blob(C.hostData, ToIntArrayRef(dimC), at::kFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }

    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpStrmm, TestStrmmTransAndLeft)
{
    Status status;
    float alpha = 2.3456;
    int64_t m = 3;
    int64_t n = 3;
    int64_t lda = 3;
    int64_t ldb = 3;
    int64_t ldc = 3;
    asdBlasSideMode_t side = AsdSip::asdBlasSideMode_t::ASDBLAS_SIDE_LEFT;
    asdBlasFillMode_t uplo = asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER;
    asdBlasOperation_t trans = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasDiagType_t diag = asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT;
    SVector<TensorDesc> inTensorDescs;     
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, m}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m, n}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[0];
    Tensor B = context.inTensors[1];
    Tensor C = context.inTensors[2];
    auto dimC = C.desc.dims;

    aclTensor *aclA = context.aclInTensors[0];
    aclTensor *aclB = context.aclInTensors[1];
    aclTensor *aclC = context.aclInTensors[2];

    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kFloat);
    at::Tensor atB = at::from_blob(B.hostData, ToIntArrayRef(B.desc.dims), at::kFloat);
    
    at::Tensor golden = StrmmGolden(side, trans, alpha, atA, atB);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeStrmmPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus =
        asdBlasStrmm(handle, side, uplo, trans, diag, m, n, alpha, aclA, lda, aclB, ldb, aclC, ldc);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);
    // CopyDeviceTensorToHostTensor(C);
    void *outputCDeviceAddr = Mki::GetStorageAddr(aclC);
    aclrtMemcpy(C.hostData, m * n * sizeof(float), outputCDeviceAddr, m * n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    asdBlasDestroy(handle);

    at::Tensor ret = at::from_blob(C.hostData, ToIntArrayRef(dimC), at::kFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }

    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}