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


at::Tensor HCgemmBatchedGolden(at::Tensor a, asdBlasOperation_t transA, at::Tensor b, asdBlasOperation_t transB)
{
    at::Tensor golden;
    at::Tensor inA = a;
    at::Tensor inB = b;

    if (transA != asdBlasOperation_t::ASDBLAS_OP_N || transB != asdBlasOperation_t::ASDBLAS_OP_N) {
        throw std::runtime_error("Invalid type of trans.");
    }

    if (a.dtype() != at::kComplexFloat) {
        inA = a.to(at::kComplexFloat);
        inB = b.to(at::kComplexFloat);
    }
    golden = at::bmm(inA, inB);
    if (a.dtype() != at::kComplexFloat) {
        golden = golden.to(a.dtype());
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

void TestHCgemm(int64_t batch, int64_t m, int64_t k, int64_t n) {

    int64_t lda = k;
    int64_t ldb = n;
    int64_t ldc = n;

    Status status;

    std::complex<op::fp16_t> alpha = std::complex<op::fp16_t>{1.0, 0.0};
    std::complex<op::fp16_t> beta = std::complex<op::fp16_t>{0.0, 0.0};
    asdBlasOperation_t transA = asdBlasOperation_t::ASDBLAS_OP_N;
    asdBlasOperation_t transB = asdBlasOperation_t::ASDBLAS_OP_N;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, m, k}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, k, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, m, n}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor a = context.inTensors[0];
    Tensor b = context.inTensors[1];
    Tensor c = context.inTensors[2];

    aclTensor *aclA = context.aclInTensors[0];
    aclTensor *aclB = context.aclInTensors[1];
    aclTensor *aclC = context.aclInTensors[2];

    at::Tensor atA = at::from_blob(a.hostData, ToIntArrayRef(a.desc.dims), at::kComplexHalf);
    at::Tensor atB = at::from_blob(b.hostData, ToIntArrayRef(b.desc.dims), at::kComplexHalf);

    at::Tensor golden = HCgemmBatchedGolden(atA, transA, atB, transB);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeHCgemmBatchedPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasHCgemmBatched(handle, transA, transB, m, n, k, alpha, aclA, lda, aclB, ldb, beta, aclC, ldc, batch);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);

    asdBlasSynchronize(handle);
    void *cDeviceAddr = Mki::GetStorageAddr(aclC);
    aclrtMemcpy(c.hostData, batch * m * n * sizeof(std::complex<op::fp16_t>),
                cDeviceAddr, batch * m * n * sizeof(std::complex<op::fp16_t>), ACL_MEMCPY_DEVICE_TO_HOST);
    at::Tensor ret = at::from_blob(c.hostData, ToIntArrayRef(c.desc.dims), at::kComplexHalf);

    at::Tensor errorIdx = at::logical_not(at::isclose(golden.flatten(), ret.flatten(), ATOL, RTOL, true));
    if ((errorIdx.sum().item().toDouble() / errorIdx.numel()) > EXTENT_OF_ERROR) {
        status = Status::FailStatus(-1, "judge not equal");
    }

    asdBlasDestroy(handle);
    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}


TEST(TestOpHCgemmBatched, TestHCgemmBatche0)
{
    TestHCgemm(15, 7, 8, 9);
}

TEST(TestOpHCgemmBatched, TestHCgemmBatche1)
{
    TestHCgemm(200, 11, 10, 9);
}
