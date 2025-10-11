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

at::Tensor SwapLast2AxesGolden(at::Tensor input)
{
    at::Tensor golden;
    golden = at::transpose(input, -1, -2);

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


bool RunSwapLast2Axes(SVector<TensorDesc> tensorDescs)
{
    bool result = true;
    aclTensorContext context;
    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    OpTestMallocInAclTensors(tensorDescs, context);

    Tensor input = context.inTensors[0];
    Tensor output = context.inTensors[1];
    aclTensor *aclIn = context.aclInTensors[0];
    aclTensor *aclOut = context.aclInTensors[1];

    uint8_t *workspace = nullptr;
    size_t lwork = 0;
    swapLast2AxesGetWorkspaceSize(lwork);
    Mki::MkiRtMemMallocDevice((void **)&workspace, lwork, MKIRT_MEM_DEFAULT);

    // 执行swaplast2axes算子
    AspbStatus blasStatus = swapLast2Axes(aclIn, aclOut, stream, workspace);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Fail to run op swapLast2Axes!";
        result = false;
        OpTestEnd(deviceId, context, stream);
        return result;
    }

    aclrtSynchronizeStream(stream);

    void *outDeviceAddr = Mki::GetStorageAddr(aclOut);
    aclrtMemcpy(output.hostData,
        output.dataSize,
        outDeviceAddr,
        output.dataSize,
        ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor atInput = at::from_blob(input.hostData, ToIntArrayRef(input.desc.dims), at::kComplexFloat);
    at::Tensor golden = SwapLast2AxesGolden(atInput);
    at::Tensor ret = at::from_blob(output.hostData, ToIntArrayRef(output.desc.dims), at::kComplexFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        ASDSIP_LOG(ERROR) << "judge not equal!";
        result = false;
    }

    OpTestEnd(deviceId, context, stream);
    return result;
}

TEST(TestOpSwapLast2Axes, TestSwapLast2AxesF32)
{
    int64_t m = 4;
    int64_t n = 6;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n, m}});

    ASSERT_EQ(RunSwapLast2Axes(inTensorDescs), true);
}

TEST(TestOpSwapLast2Axes, TestSwapLast2AxesF32Batch)
{
    int64_t m = 4;
    int64_t n = 6;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1, m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1, n, m}});

    ASSERT_EQ(RunSwapLast2Axes(inTensorDescs), true);
}

TEST(TestOpSwapLast2Axes, TestSwapLast2AxesF32OutAndInNotEqual)
{
    int64_t m = 4;
    int64_t n = 6;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m, m}});

    ASSERT_EQ(RunSwapLast2Axes(inTensorDescs), false);
}

TEST(TestOpSwapLast2Axes, TestSwapLast2AxesF32BatchNotVaild)
{
    int64_t m = 4;
    int64_t n = 6;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1, 2, m, n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1, 2, n, m}});

    ASSERT_EQ(RunSwapLast2Axes(inTensorDescs), false);
}