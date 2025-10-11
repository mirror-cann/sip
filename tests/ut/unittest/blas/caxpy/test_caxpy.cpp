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

using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float EXTENT_OF_ERROR = 0.001;

at::Tensor CaxpyGolden(std::complex<float> alpha, at::Tensor x, at::Tensor y)
{
    at::Tensor golden;
    auto atAlpha = torch::complex(torch::tensor(alpha.real()), torch::tensor(alpha.imag()));
    golden = atAlpha * x + y;
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

struct CaxpyUTParam{
    SVector<TensorDesc> inTensorDescs;
    int64_t n;
    int64_t incx;
    int64_t incy;
    std::complex<float> alpha;
};

bool RunCaxpy(struct CaxpyUTParam param, bool set_plan_handle_null=false) 
{
    int64_t n = param.n;
    int64_t incx = param.incx;
    int64_t incy = param.incy;
    std::complex<float> alpha = param.alpha;
    SVector<TensorDesc> inTensorDescs = param.inTensorDescs;

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

    size_t lwork = 0;
    uint8_t *buffer = nullptr;

    AspbStatus status;

    if (set_plan_handle_null == false)
    {
        status = asdBlasMakeCaxpyPlan(handle);
    }
    else 
    {
        status = asdBlasMakeCaxpyPlan(nullptr);
    }
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to create plan.", return false);

    asdBlasGetWorkspaceSize(handle, lwork);
    MkiRtMemMallocDevice((void **)&buffer, lwork, MKIRT_MEM_DEFAULT);
    asdBlasSetWorkspace(handle, buffer);
    asdBlasSetStream(handle, stream);

    status = asdBlasCaxpy(handle, n, alpha, aclX, incx, aclY, incy);
    ASDSIP_CHECK(status == AsdSip::ErrorType::ACL_SUCCESS, "Fail to execute the caxpy op.", return false);

    asdBlasSynchronize(handle);
    asdBlasDestroy(handle);

    // golden
    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor atY = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexFloat);

    at::Tensor golden = CaxpyGolden(alpha, atX, atY);

    // out tensor data device to host
    void* inputYDeviceAddr = Mki::GetStorageAddr(aclY);
    aclrtMemcpy(y.hostData, n * sizeof(std::complex<float>), inputYDeviceAddr,
                    n * sizeof(std::complex<float>), ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(y.hostData, ToIntArrayRef(y.desc.dims), at::kComplexFloat);
    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        ASDSIP_LOG(ERROR) << "judge not equal.";
        OpTestEnd(deviceId, context, stream);
        return false;
    }
    OpTestEnd(deviceId, context, stream);
    return true;
}

TEST(TestOpCaxpy, TestCaxpyF32)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    ASSERT_EQ(RunCaxpy({inTensorDescs, n, incx, incy, alpha}), true);
}

TEST(TestOpCaxpy, TestCaxpyF32LongDims)
{
    int64_t n = 128;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, n, incx, incy, alpha}), true);
}

// invalid case n <= 0
TEST(TestOpCaxpy, TestCaxpyInvalidN01)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, -1, incx, incy, alpha}), false);
}

// invalid case n > uint32_max
TEST(TestOpCaxpy, TestCaxpyInvalidN02)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, (UINT32_MAX + 1), incx, incy, alpha}), false);
}

// invalid case n 与 x 的shape not equal
TEST(TestOpCaxpy, TestCaxpyInvalidXShape)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n + 16}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, n, incx, incy, alpha}), false);
}

// invalid case n 与 y 的shape not equal
TEST(TestOpCaxpy, TestCaxpyInvalidYShape)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n + 16}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, n, incx, incy, alpha}), false);
}

// invalid case not supported x dtype
TEST(TestOpCaxpy, TestCaxpyInvalidXDtype)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, n, incx, incy, alpha}), false);
}

// invalid case not supported x dtype
TEST(TestOpCaxpy, TestCaxpyInvalidYDtype)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, n, incx, incy, alpha}), false);
}

// invalid case not supported x dtype
TEST(TestOpCaxpy, TestCaxpyInvalidIncx)
{
    int64_t n = 16;
    int64_t incx = -1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, n, incx, incy, alpha}), true);
}

// invalid case not supported x dtype
TEST(TestOpCaxpy, TestCaxpyInvalidIncy)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = -1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, n, incx, incy, alpha}), true);
}

// invalid case null handle
TEST(TestOpCaxpy, TestCaxpyNullHandle)
{
    int64_t n = 16;
    int64_t incx = 1;
    int64_t incy = 1;
    std::complex<float> alpha = (std::complex<float>) {1.0, 1.0};

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {n}});
    aclTensorContext context;
    ASSERT_EQ(RunCaxpy({inTensorDescs, n, incx, incy, alpha}, true), false);
}