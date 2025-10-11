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
#include "test_util/util.cpp"
#include "base_api.h"
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

at::Tensor AsdMulGolden(at::Tensor inputX, at::Tensor inputY)
{
    at::Tensor golden;
    golden = inputX * inputY;
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


bool RunAsdMul(SVector<TensorDesc> tensorDescs)
{
    bool result = true;
    aclTensorContext context;
    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    OpTestMallocInAclTensors(tensorDescs, context);
    Tensor inputX = context.inTensors[0];
    Tensor inputY = context.inTensors[1];
    Tensor output = context.inTensors[2];
    aclTensor *aclInX = context.aclInTensors[0];
    aclTensor *aclInY = context.aclInTensors[1];
    aclTensor *aclOut = context.aclInTensors[2];

    // run op asdMul
    AspbStatus blasStatus = asdMul(inputX.Numel(), aclInX, aclInY, aclOut, stream);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Fail to run op asdMul!";
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

    at::Tensor atInputX = at::from_blob(inputX.hostData, ToIntArrayRef(inputX.desc.dims), DTypeMapper::ConvertToTorchDtype(inputX.desc.dtype));
    at::Tensor atInputY = at::from_blob(inputY.hostData, ToIntArrayRef(inputY.desc.dims), DTypeMapper::ConvertToTorchDtype(inputY.desc.dtype));
    at::Tensor golden = AsdMulGolden(atInputX, atInputY);
    at::Tensor ret = at::from_blob(output.hostData, ToIntArrayRef(output.desc.dims), DTypeMapper::ConvertToTorchDtype(output.desc.dtype));

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        ASDSIP_LOG(ERROR) << "judge not equal!";
        result = false;
    }

    OpTestEnd(deviceId, context, stream);
    return result;
}

TEST(TestOpAsdMul, TestAsdMulC64)
{
    int64_t m = 10;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m}});

    ASSERT_EQ(RunAsdMul(inTensorDescs), true);
}

TEST(TestOpAsdMul, TestAsdMulC32)
{
    int64_t m = 10;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {m}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {m}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {m}});

    ASSERT_EQ(RunAsdMul(inTensorDescs), true);
}

TEST(TestOpAsdMul, TestAsdMulF32)
{
    int64_t m = 4;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {m}});
    ASSERT_EQ(RunAsdMul(inTensorDescs), false);
}

TEST(TestOpAsdMul, TestAsdMulC64InAndOutNumelNotEqual)
{
    int64_t m = 4;
    int64_t n = 6;

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {m}});

    ASSERT_EQ(RunAsdMul(inTensorDescs), false);
}