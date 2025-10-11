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

at::Tensor ScasumGolden(at::Tensor x)
{
    at::Tensor golden;
    golden = at::sum(at::abs(torch::real(x)), -1) + at::sum(at::abs(torch::imag(x)), -1);
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

bool RunScasum(SVector<TensorDesc> inTensorDescs, int64_t n, int64_t incx)
{
    AspbStatus status;
    aclTensorContext context;

    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    OpTestMallocInAclTensors(inTensorDescs, context);
    Tensor x = context.inTensors[0];
    Tensor y = context.inTensors[1];
    aclTensor* aclX = context.aclInTensors[0];
    aclTensor* aclY = context.aclInTensors[1];

    asdBlasHandle handle;
    asdBlasCreate(handle);

    status = asdBlasCreate(handle);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to create handle.", return false);

    status = asdBlasMakeAsumPlan(handle);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to create plan.", return false);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    status = asdBlasGetWorkspaceSize(handle, lwork);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to get workspace.", return false);

    Mki::MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);

    status = asdBlasSetWorkspace(handle, buffer);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to set workspace.", return false);

    status = asdBlasSetStream(handle, stream);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to set stream.", return false);

    // 执行sasum算子
    status = asdBlasScasum(handle, n, aclX, incx, aclY);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to execute the sasum op.", return false);

    status = asdBlasSynchronize(handle);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to synchronize.", return false);

    void* inputYDeviceAddr = Mki::GetStorageAddr(aclY);
    aclrtMemcpy(y.hostData, sizeof(float), inputYDeviceAddr,
                    sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);

    status = asdBlasDestroy(handle);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to destroy handle.", return false);

    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor golden = ScasumGolden(atX);
    at::Tensor ret = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kFloat);
    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        ASDSIP_LOG(ERROR) << "judge not equal.";
        OpTestEnd(deviceId, context, stream);
        return false;
    }

    OpTestEnd(deviceId, context, stream);
    return true;
}

TEST(TestOpScasum, TestScasumF32)
{
    int64_t n = 16;
    int64_t incx = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1}});
   
    ASSERT_EQ(RunScasum(inTensorDescs, n, incx), true);
}

TEST(TestOpScasum, TestScasumInINT32)
{
    int64_t n = 16;
    int64_t incx = 1;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1}});

    ASSERT_EQ(RunScasum(inTensorDescs, n, incx), false);
}

TEST(TestOpScasum, TestScasumOutINT32)
{
    int64_t n = 16;
    int64_t incx = 1;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {1}});

    ASSERT_EQ(RunScasum(inTensorDescs, n, incx), false);
}

TEST(TestOpScasum, TestScasumInvalidInShape)
{
    int64_t n = 16;
    int64_t incx = 1;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1}});

    ASSERT_EQ(RunScasum(inTensorDescs, 2 * n, incx), false);
}

TEST(TestOpScasum, TestScasumInvalidOutShape)
{
    int64_t n = 16;
    int64_t incx = 1;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1 * 2}});

    ASSERT_EQ(RunScasum(inTensorDescs, n, incx), false);
}

TEST(TestOpScasum, TestScasumInvalidIncx)
{
    int64_t n = 16;
    int64_t incx = -2;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1}});

    ASSERT_EQ(RunScasum(inTensorDescs, n, incx), true);
}

TEST(TestOpScasum, TestScasumInvalidInFormat)
{
    int64_t n = 16;
    int64_t incx = 1;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_NHWC, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1}});

    ASSERT_EQ(RunScasum(inTensorDescs, n, incx), false);
}

TEST(TestOpScasum, TestScasumInvalidOutFormat)
{
    int64_t n = 16;
    int64_t incx = 1;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_NHWC, {1}});

    ASSERT_EQ(RunScasum(inTensorDescs, n, incx), false);
}

TEST(TestOpScasum, TestScasumInvalidN01)
{
    int64_t n = UINT32_MAX + 1;
    int64_t incx = 1;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {16}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1}});

    ASSERT_EQ(RunScasum(inTensorDescs, n, incx), false);
}

TEST(TestOpScasum, TestScasumInvalidN02)
{
    int64_t n = -1;
    int64_t incx = 1;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {16}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1}});

    ASSERT_EQ(RunScasum(inTensorDescs, n, incx), false);
}
