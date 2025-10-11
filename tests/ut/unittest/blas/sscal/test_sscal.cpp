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
#include "test_common.h"
#include "test_util/util.cpp"
#include "blas_api.h"
#include "utils/mem_base_inner.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float EXTENT_OF_ERROR = 0.001;

at::Tensor SscalGolden(float alpha, at::Tensor x)
{
    at::Tensor golden;
    golden = alpha * x;
    return golden;
}

int AclInit(int32_t deviceId, aclrtStream *stream)
{
    auto ret = aclrtSetDevice(deviceId);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtSetDevice failed. ERROR: %d\n", ret); return ret);
    ret = aclrtCreateStream(stream);
    CHECK_RET(ret == ::ACL_SUCCESS, LOG_PRINT("aclrtCreateStream failed. ERROR: %d\n", ret); return ret);
    return 0;
}

}  // namespace

bool RunSscal(SVector<TensorDesc> inTensorDescs, int64_t n, int64_t incx, float alpha)
{
    aclTensorContext context;

    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    OpTestMallocInAclTensors(inTensorDescs, context);
    Tensor x = context.inTensors[0];
    aclTensor *aclX = context.aclInTensors[0];

    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kFloat);
    at::Tensor golden = SscalGolden(alpha, atX);

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    asdBlasMakeCalPlan(handle);
    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    AspbStatus blasStatus = asdBlasSscal(handle, n, alpha, aclX, incx);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "run ops failed!";
        return false;
    }

    asdBlasSynchronize(handle);
    void *xDeviceAddr = Mki::GetStorageAddr(aclX);
    aclrtMemcpy(x.hostData, n * sizeof(float), xDeviceAddr, n * sizeof(float), ACL_MEMCPY_DEVICE_TO_HOST);
    asdBlasDestroy(handle);

    at::Tensor ret = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        ASDSIP_LOG(ERROR) << "judge not equal!";
        OpTestEnd(deviceId, context, stream);
        return false;
    }

    OpTestEnd(deviceId, context, stream);
    return true;
}

TEST(TestOpSscal, TestSscalF32)
{
    int64_t n = 16;
    int64_t incx = 1;
    float alpha = 2.0;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n}});

    auto ret = RunSscal(inTensorDescs, n, incx, alpha);
    ASSERT_EQ(ret, true);
}

TEST(TestOpSscal, TestSscalF32Unaligned)
{
    int64_t n = 653;
    int64_t incx = 1;
    float alpha = 2.0;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n}});

    auto ret = RunSscal(inTensorDescs, n, incx, alpha);
    ASSERT_EQ(ret, true);
}

TEST(TestOpSscal, TestSscalF32InvalidShape)
{
    int64_t n = 16;
    int64_t incx = 1;
    float alpha = 2.0;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n + 10}});

    auto ret = RunSscal(inTensorDescs, n, incx, alpha);
    ASSERT_EQ(ret, false);
}

TEST(TestOpSscal, TestSscalF32InvalidIncx)
{
    int64_t n = 16;
    int64_t incx = -1;
    float alpha = 2.0;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {n}});

    auto ret = RunSscal(inTensorDescs, n, incx, alpha);
    ASSERT_EQ(ret, true);
}

TEST(TestOpSscal, TestSscalInt32)
{
    int64_t n = 16;
    int64_t incx = 1;
    float alpha = 2.0;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {n}});

    auto ret = RunSscal(inTensorDescs, n, incx, alpha);
    ASSERT_EQ(ret, false);
}