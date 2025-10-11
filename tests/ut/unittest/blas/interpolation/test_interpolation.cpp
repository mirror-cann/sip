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
#include "domain/rs_api.h"
#include "interp_api.h"
#include "utils/mem_base_inner.h"
#include "sinc_interp.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"

using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;
constexpr float EXTENT_OF_ERROR = 0.001;

uint8_t InterpolationGolden(const at::Tensor &inTensor, const at::Tensor &posTensor, const int64_t batch,
                          const int64_t signalLength, const int64_t interpLength, at::Tensor &outTensor)
{
    complex<float> *inSignal = (complex<float> *)inTensor.data_ptr();
    double *intpPos = (double *)posTensor.data_ptr();
    complex<float> *outSignal = (complex<float> *)outTensor.data_ptr();
    for (int i = 0; i < batch; i++) {
        BaseBndSinc32by16_float(inSignal + i * signalLength, signalLength, 0, 1, intpPos + i * interpLength,
                                interpLength, outSignal + i * interpLength);
    }

    return 0;
}

at::Tensor ComplexMatDotGolden(at::Tensor matX, at::Tensor matY)
{
    at::Tensor golden = torch::matmul(matX, matY);

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

Status runComplex64(int64_t batch, int64_t nRs, int64_t totalSubcarrier, int64_t nSingal)
{
    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, nSingal - nRs, nRs}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, nRs, totalSubcarrier}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, nSingal - nRs, totalSubcarrier}});
    aclTensorContext context;

    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor coeff = context.inTensors[0];
    Tensor x = context.inTensors[1];
    Tensor out = context.inTensors[2];

    aclTensor *aclCoeff = context.aclInTensors[0];
    aclTensor *aclX = context.aclInTensors[1];
    aclTensor *aclOut = context.aclInTensors[2];

    at::Tensor atCoeff = at::from_blob(coeff.hostData, ToIntArrayRef(coeff.desc.dims), at::kComplexFloat);
    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexFloat);
    at::Tensor golden = ComplexMatDotGolden(atCoeff, atX);

    void* workspace = nullptr;
    size_t lwork = 0;
    asdInterpWithCoeffGetWorkspaceSize(lwork);
    MkiRtMemMallocDevice((void **)&workspace, lwork, MKIRT_MEM_DEFAULT);

    AspbStatus res = asdInterpWithCoeff(aclX, aclCoeff, aclOut, stream, workspace);
    if (res != AsdSip::ErrorType::ACL_SUCCESS) {
        std::cout << "run ops failed :"<< res << std::endl;
        status = Status::FailStatus(-1, "run ops failed!");
        return status;
    }

    MkiRtStreamSynchronize(stream);
    void *outPutDeviceAddr = Mki::GetStorageAddr(aclOut);
    aclrtMemcpy(out.hostData,
        batch * (nSingal - nRs) * totalSubcarrier * sizeof(std::complex<float>),
        outPutDeviceAddr,
        batch * (nSingal - nRs) * totalSubcarrier * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(out.hostData, ToIntArrayRef(out.desc.dims), at::kComplexFloat);

    if (!at::allclose(golden, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
        std::cout << "golden :"<< golden << std::endl;
        std::cout << "ret :"<< ret << std::endl;
    }

    OpTestEnd(deviceId, context, stream);
    return status;
}

Status runComplex32(int64_t batch, int64_t nRs, int64_t totalSubcarrier, int64_t nSingal)
{
    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, nSingal - nRs, nRs}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, nRs, totalSubcarrier}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX32, TENSOR_FORMAT_ND, {batch, nSingal - nRs, totalSubcarrier}});
    aclTensorContext context;

    int deviceId = 0;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor coeff = context.inTensors[0];
    Tensor x = context.inTensors[1];
    Tensor out = context.inTensors[2];

    aclTensor *aclCoeff = context.aclInTensors[0];
    aclTensor *aclX = context.aclInTensors[1];
    aclTensor *aclOut = context.aclInTensors[2];

    at::Tensor atCoeff = at::from_blob(coeff.hostData, ToIntArrayRef(coeff.desc.dims), at::kComplexHalf);
    at::Tensor atX = at::from_blob(x.hostData, ToIntArrayRef(x.desc.dims), at::kComplexHalf);
    at::Tensor golden = ComplexMatDotGolden(atCoeff.to(at::kComplexFloat), atX.to(at::kComplexFloat));

    void* workspace = nullptr;
    size_t lwork = 0;
    asdInterpWithCoeffGetWorkspaceSize(lwork);
    MkiRtMemMallocDevice((void **)&workspace, lwork, MKIRT_MEM_DEFAULT);

    AspbStatus res = asdInterpWithCoeff(aclX, aclCoeff, aclOut, stream, workspace);
    if (res != AsdSip::ErrorType::ACL_SUCCESS) {
        std::cout << "run ops failed :"<< res << std::endl;
        status = Status::FailStatus(-1, "run ops failed!");
        return status;
    }

    MkiRtStreamSynchronize(stream);
    void *outPutDeviceAddr = Mki::GetStorageAddr(aclOut);
    aclrtMemcpy(out.hostData,
        batch * (nSingal - nRs) * totalSubcarrier * sizeof(float),
        outPutDeviceAddr,
        batch * (nSingal - nRs) * totalSubcarrier * sizeof(float),
        ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(out.hostData, ToIntArrayRef(out.desc.dims), at::kComplexHalf);

    if (!at::allclose(golden.to(at::kComplexHalf), ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
        std::cout << "golden :"<< golden << std::endl;
        std::cout << "ret :"<< ret << std::endl;
    }

    OpTestEnd(deviceId, context, stream);
    return status;
}

}

TEST(TestOpInterpolation, TestInterpolationF32)
{
    int64_t batch = 1;
    int64_t len = 64;
    int64_t num = 64;
    int interpNum = 16;
    int quantNum = 32;
    int tabSize = (quantNum * 2) * (interpNum * 2 + 8) * 4; 

    Status status;
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, len}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {1, tabSize}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT32, TENSOR_FORMAT_ND, {batch, num}});
    inTensorDescs.push_back({TENSOR_DTYPE_INT16, TENSOR_FORMAT_ND, {batch, num}});
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, num}});
    aclTensorContext context;

    int deviceId = 0;
    std::cout << "begin create stream" << std::endl;
    aclrtStream stream;
    AclInit(deviceId, &stream);

    std::cout << "begin OpTestMallocInAclTensors" << std::endl;
    OpTestMallocInAclTensors(inTensorDescs, context);

    Tensor in = context.inTensors[0];
    Tensor tab = context.inTensors[1];
    Tensor pos = context.inTensors[2];
    Tensor posTab = context.inTensors[3];
    Tensor out = context.inTensors[4];

    aclTensor *aclIn = context.aclInTensors[0];
    aclTensor *aclTab = context.aclInTensors[1];
    aclTensor *aclPos = context.aclInTensors[2];
    aclTensor *aclPosTab = context.aclInTensors[3];
    aclTensor *aclOut = context.aclInTensors[4];

    at::Tensor atIn = at::from_blob(in.hostData, ToIntArrayRef(in.desc.dims), at::kComplexFloat);
    at::Tensor atPos = at::from_blob(pos.hostData, ToIntArrayRef(pos.desc.dims), at::kFloat);
    at::Tensor atOut = at::from_blob(out.hostData, ToIntArrayRef(out.desc.dims), at::kComplexFloat);

    InterpolationGolden(atIn, atPos, batch, len, num, atOut);

    void* workspace = nullptr;
    size_t lwork = 0;
    rsInterpolationBySincGetWorkspaceSize(lwork);
    MkiRtMemMallocDevice((void **)&workspace, lwork, MKIRT_MEM_DEFAULT);

    AspbStatus blasStatus = rsInterpolationBySinc(aclIn, aclTab, aclPos, aclPosTab, aclOut, 16, 32, num, stream, workspace);
    if (blasStatus != AsdSip::ErrorType::ACL_SUCCESS) {
        status = Status::FailStatus(-1, "run ops failed!");
    }
    ASSERT_EQ(status.Ok(), true);
    MkiRtStreamSynchronize(stream);
    // CopyOutTensorToHost(out);
    void *outPutDeviceAddr = Mki::GetStorageAddr(aclOut);
    aclrtMemcpy(out.hostData,
        batch * num * sizeof(std::complex<float>),
        outPutDeviceAddr,
        batch * num * sizeof(std::complex<float>),
        ACL_MEMCPY_DEVICE_TO_HOST);

    at::Tensor ret = at::from_blob(out.hostData, ToIntArrayRef(out.desc.dims), at::kComplexFloat);

    if (!at::allclose(atOut, ret, EXTENT_OF_ERROR, EXTENT_OF_ERROR)) {
        status = Status::FailStatus(-1, "judge not equal");
    }

    OpTestEnd(deviceId, context, stream);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpInterpolation, TestInterpWithCoeff_C64_01)
{
    Status status = runComplex64(1, 2, 12, 14);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpInterpolation, TestInterpWithCoeff_C64_02)
{
    Status status = runComplex64(20, 2, 12, 14);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpInterpolation, TestInterpWithCoeff_C64_03)
{
    Status status = runComplex64(25, 2, 12, 14);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpInterpolation, TestInterpWithCoeff_C64_04)
{
    Status status = runComplex64(13, 2, 123, 14);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpInterpolation, TestInterpWithCoeff_C32_01)
{
    Status status = runComplex32(1, 4, 12, 14);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpInterpolation, TestInterpWithCoeff_C32_02)
{
    Status status = runComplex32(20, 4, 12, 14);
    ASSERT_EQ(status.Ok(), true);
}

TEST(TestOpInterpolation, TestInterpWithCoeff_C32_03)
{
    Status status = runComplex32(13, 4, 123, 14);
    ASSERT_EQ(status.Ok(), true);
}