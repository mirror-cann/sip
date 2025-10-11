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

using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float EXTENT_OF_ERROR = 0.001;

at::Tensor CgemmGolden(asdBlasOperation_t transa, asdBlasOperation_t transb, std::complex<float> alpha, 
                    std::complex<float> beta, at::Tensor A, at::Tensor B, at::Tensor C)
{
    auto atAlpha = torch::complex(torch::tensor(alpha.real()), torch::tensor(alpha.imag()));
    auto atBeta = torch::complex(torch::tensor(beta.real()), torch::tensor(beta.imag()));
    if (transa == asdBlasOperation_t::ASDBLAS_OP_N) {
        A = at::transpose(A, 1, 0);
    } else if (transa == asdBlasOperation_t::ASDBLAS_OP_C) {
        A = at::conj(A);
    }
    if (transb == asdBlasOperation_t::ASDBLAS_OP_N) {
        B = at::transpose(B, 1, 0);
    } else if (transb == asdBlasOperation_t::ASDBLAS_OP_C) {
        B = at::conj(B);
    }
    C =  at::transpose(C, 1, 0);
    at::Tensor golden;
    golden = atAlpha * at::matmul(A, B) + atBeta * C;
    golden = at::transpose(golden, 1, 0);
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


TEST(TestOpCgemm, TestCgemmF32NoTrans)
{
    Status status;
    asdBlasOperation_t transa = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasOperation_t transb = asdBlasOperation_t::ASDBLAS_OP_N;
    int64_t m = 2;
    int64_t n = 2;
    int64_t k = 2;
    std::complex<float> alpha = {1.0, 2.0};
    std::complex<float> beta = {1.0, 2.0};
    int64_t lda = 2;
    int64_t ldb = 2;
    int64_t ldc = 2;


    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, k}}); // A
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {k, n}}); // B
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}}); // C

    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[0];
    Tensor B = context.inTensors[1];
    Tensor C = context.inTensors[2];

    aclTensor* aclA = context.aclInTensors[0];
    aclTensor* aclB = context.aclInTensors[1];
    aclTensor* aclC = context.aclInTensors[2];
    
    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kComplexFloat);
    at::Tensor atB = at::from_blob(B.hostData, ToIntArrayRef(B.desc.dims), at::kComplexFloat);
    at::Tensor atC = at::from_blob(C.hostData, ToIntArrayRef(C.desc.dims), at::kComplexFloat);

    at::Tensor golden = CgemmGolden(transa, transb, alpha, beta, atA, atB, atC);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeCgemmPlan(handle, transa, transb, m, n, k, lda, ldb, ldc);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasCgemm(handle, transa, transb, m, n, k, alpha, aclA, lda, aclB, ldb, beta, aclC, ldc);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);
    asdBlasSynchronize(handle);
    // CopyDeviceTensorToHostTensor(C);
    void* inputCDeviceAddr = Mki::GetStorageAddr(aclC);

    aclrtMemcpy(C.hostData, m * n * sizeof(std::complex<float>), inputCDeviceAddr,
                    m * n * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(C.hostData, ToIntArrayRef(C.desc.dims), at::kComplexFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    asdBlasDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpCgemm, TestCgemmF32Conj)
{
    Status status;
    asdBlasOperation_t transa = asdBlasOperation_t::ASDBLAS_OP_C;
    asdBlasOperation_t transb = asdBlasOperation_t::ASDBLAS_OP_C;
    int64_t m = 2;
    int64_t n = 2;
    int64_t k = 2;
    std::complex<float> alpha = {1.0, 2.0};
    std::complex<float> beta = {1.0, 2.0};
    int64_t lda = 2;
    int64_t ldb = 2;
    int64_t ldc = 2;


    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, k}}); // A
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {k, n}}); // B
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}}); // C

    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor A = context.inTensors[0];
    Tensor B = context.inTensors[1];
    Tensor C = context.inTensors[2];

    aclTensor* aclA = context.aclInTensors[0];
    aclTensor* aclB = context.aclInTensors[1];
    aclTensor* aclC = context.aclInTensors[2];
    
    at::Tensor atA = at::from_blob(A.hostData, ToIntArrayRef(A.desc.dims), at::kComplexFloat);
    at::Tensor atB = at::from_blob(B.hostData, ToIntArrayRef(B.desc.dims), at::kComplexFloat);
    at::Tensor atC = at::from_blob(C.hostData, ToIntArrayRef(C.desc.dims), at::kComplexFloat);

    at::Tensor golden = CgemmGolden(transa, transb, alpha, beta, atA, atB, atC);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeCgemmPlan(handle, transa, transb, m, n, k, lda, ldb, ldc);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasCgemm(handle, transa, transb, m, n, k, alpha, aclA, lda, aclB, ldb, beta, aclC, ldc);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);
    asdBlasSynchronize(handle);
    // CopyDeviceTensorToHostTensor(C);
    void* inputCDeviceAddr = Mki::GetStorageAddr(aclC);

    aclrtMemcpy(C.hostData, m * n * sizeof(std::complex<float>), inputCDeviceAddr,
                    m * n * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(C.hostData, ToIntArrayRef(C.desc.dims), at::kComplexFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }
    asdBlasDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}