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

at::Tensor CdotcGolden(at::Tensor x, at::Tensor y)
{
    at::Tensor golden;
    x = at::conj(x);
    golden = at::dot(x, y);
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


struct CdotcUTParam{
    SVector<TensorDesc> inTensorDescs;
    int64_t n;
    int64_t incx;
    int64_t incy;
};

bool RunCdotc(struct CdotcUTParam param, bool set_plan_handle_null=false)
{
    int64_t n = param.n;
    int64_t incx = param.incx;
    int64_t incy = param.incy;

    SVector<TensorDesc> inTensorDescs = param.inTensorDescs;

    aclTensorContext context;

    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor x = context.inTensors[0];
    Tensor y = context.inTensors[1];
    Tensor result = context.inTensors[2];

    aclTensor *aclX = context.aclInTensors[0];
    aclTensor *aclY = context.aclInTensors[1];
    aclTensor *aclResult = context.aclInTensors[2];

    asdBlasHandle handle;
    asdBlasCreate(handle);

    size_t lwork = 0;
    uint8_t *buffer = nullptr;
    AspbStatus status;

    if (set_plan_handle_null == false)
    {
        status = asdBlasMakeDotPlan(handle);
    }
    else
    {
        status = asdBlasMakeDotPlan(nullptr);
    }
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to create plan.", return false);

    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    status = asdBlasCdotc(handle, n, aclX, incx, aclY, incy, aclResult);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to execute the ccopy op.", return false);

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor atY = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    at::Tensor golden = CdotcGolden(atX, atY);

    void *resultAddr = Mki::GetStorageAddr(aclResult);
    aclrtMemcpy(result.hostData, sizeof(std::complex<float>), resultAddr,
        sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(result.hostData, ToIntArrayRef(result.desc.dims), at::kComplexFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        ASDSIP_LOG(ERROR) << "judge not equal.";
        OpTestEnd(deviceId, context, stream);
        return false;
    }

    OpTestEnd(deviceId, context, stream);
    return true;

}


TEST(TestOpCdotc, TestCdotcF32)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), true);
}

TEST(TestOpCdotc, TestCdotcF32LongDims)
{
    int64_t n = 1024;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), true);
}


TEST(TestOpCdotc, TestCdotcF32Not32)
{
    int64_t n = 32 * 12 + 3;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), true);
}


TEST(TestOpCdotc, TestCdotcF32InSizeNotEqualX)
{
    int64_t n = 32 * 12 + 3;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n + 15}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), false);
}


TEST(TestOpCdotc, TestCdotcF32InSizeNotEqualY)
{
    int64_t n = 32 * 12 + 3;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n + 13}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), false);
}

TEST(TestOpCdotc, TestCdotcF32OutSizeNotOne)
{
    int64_t n = 32 * 12 + 3;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {3}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), false);
}


TEST(TestOpCdotc, TestCdotcF32DtypeNotEqualX)
{
    int64_t n = 32 * 12 + 3;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), false);
}


TEST(TestOpCdotc, TestCdotcF32DtypeNotEqualY)
{
    int64_t n = 32 * 12 + 3;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), false);
}


TEST(TestOpCdotc, TestCdotcF32DtypeNotEqualRes)
{
    int64_t n = 32 * 12 + 3;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), false);
}


TEST(TestOpCdotc, TestCdotcF32InvalidIncx)
{
    int64_t n = 16;
    int64_t incx = -1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), true);
}


TEST(TestOpCdotc, TestCdotcF32InvalidIncy)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = -1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}), true);
}

TEST(TestOpCdotc, TestCdotcF32NullHandle)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {1}});
    aclTensorContext context;

    ASSERT_EQ(RunCdotc({inTensorDescs, n, incx, incy}, true), false);
}