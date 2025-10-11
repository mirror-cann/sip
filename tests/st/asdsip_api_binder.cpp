/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <string>
#include <iostream>
#include <torch/extension.h>
#include <torch_npu/csrc/core/npu/NPUStream.h>

#undef SUCCESS
#undef FAILED

#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include "utils/mem_base_inner.h"
#include "base_api.h"
#include "blas_api.h"
#include "filter_api.h"
#include "fft_api.h"
#include "interp_api.h"
#include "domain/rs_api.h"

#include "acl/acl.h"
#include "aclnn/opdev/fp16_t.h"
#include "log/log.h"
#include "mki/utils/platform/platform_info.h"
#include <chrono>
#include "sinc_interp.h"

using namespace std::chrono;

static std::map<at::ScalarType, Mki::TensorDType> DTYPE_MAP = {
    {torch::ScalarType::Byte, Mki::TENSOR_DTYPE_UINT8},
    {torch::ScalarType::Char, Mki::TENSOR_DTYPE_INT8},
    {torch::ScalarType::Half, Mki::TENSOR_DTYPE_FLOAT16},
    {torch::ScalarType::Float, Mki::TENSOR_DTYPE_FLOAT},
    {torch::ScalarType::Int, Mki::TENSOR_DTYPE_INT32},
    {torch::ScalarType::Long, Mki::TENSOR_DTYPE_INT64},
    {torch::ScalarType::BFloat16, Mki::TENSOR_DTYPE_BF16},
    {torch::ScalarType::ComplexFloat, Mki::TENSOR_DTYPE_COMPLEX64},
    {torch::ScalarType::ComplexDouble, Mki::TENSOR_DTYPE_COMPLEX128},
    {torch::ScalarType::ComplexHalf, Mki::TENSOR_DTYPE_COMPLEX32}};

static std::map<char, AsdSip::asdBlasSideMode_t> SIDE_MODE_MAP = {
    {'L', AsdSip::asdBlasSideMode_t::ASDBLAS_SIDE_LEFT},
    {'l', AsdSip::asdBlasSideMode_t::ASDBLAS_SIDE_LEFT},
    {'R', AsdSip::asdBlasSideMode_t::ASDBLAS_SIDE_RIGHT},
    {'r', AsdSip::asdBlasSideMode_t::ASDBLAS_SIDE_RIGHT}};

static std::map<char, AsdSip::asdBlasOperation_t> OPERATION_MAP = {
    {'N', AsdSip::asdBlasOperation_t::ASDBLAS_OP_N},
    {'n', AsdSip::asdBlasOperation_t::ASDBLAS_OP_N},
    {'T', AsdSip::asdBlasOperation_t::ASDBLAS_OP_T},
    {'t', AsdSip::asdBlasOperation_t::ASDBLAS_OP_T},
    {'C', AsdSip::asdBlasOperation_t::ASDBLAS_OP_C},
    {'c', AsdSip::asdBlasOperation_t::ASDBLAS_OP_C}};

static std::map<char, AsdSip::asdBlasFillMode_t> FILL_MODE_MAP = {
    {'L', AsdSip::asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER},
    {'l', AsdSip::asdBlasFillMode_t::ASDBLAS_FILL_MODE_LOWER},
    {'U', AsdSip::asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER},
    {'u', AsdSip::asdBlasFillMode_t::ASDBLAS_FILL_MODE_UPPER},
    {'F', AsdSip::asdBlasFillMode_t::ASDBLAS_FILL_MODE_FULL},
    {'f', AsdSip::asdBlasFillMode_t::ASDBLAS_FILL_MODE_FULL}};

static std::map<char, AsdSip::asdBlasDiagType_t> DIAG_TYPE_MAP = {
    {'N', AsdSip::asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT},
    {'n', AsdSip::asdBlasDiagType_t::ASDBLAS_DIAG_NON_UNIT},
    {'U', AsdSip::asdBlasDiagType_t::ASDBLAS_DIAG_UNIT},
    {'u', AsdSip::asdBlasDiagType_t::ASDBLAS_DIAG_UNIT}};

static std::map<std::string, AsdSip::asdConvolveMode_t> CONVOLVE_MODE_MAP = {
    {"full", AsdSip::asdConvolveMode_t::ASD_CONVOLVE_FULL},
    {"same", AsdSip::asdConvolveMode_t::ASD_CONVOLVE_SAME},
    {"valid", AsdSip::asdConvolveMode_t::ASD_CONVOLVE_VALID}};

static int32_t global_devId = 0;

void *getCurrentStream()
{
    Mki::MkiRtDeviceGetCurrent(&global_devId);
    void *stream = c10_npu::getCurrentNPUStream(global_devId).stream();
    return stream;
}

using namespace Mki;
using namespace AsdSip;

// void _setGlobalWorkspace(uint64_t bufferSize = 256*1024*1024) {
//     if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_310B) {
//         MkiRtMemMallocDevice((void **)&buffer, bufferSize, MKIRT_MEM_DEFAULT);
//         setGlobalWorkspace(buffer, bufferSize);
//     }

// }

// void _resetGlobalWorkspace() {
//     if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_310B) {
//         MkiRtMemFreeDevice((void *)buffer);
//     }
// }

void _setGlobalWorkspace(uint64_t bufferSize = 256 * 1024 * 1024)
{
    return;
}

void _resetGlobalWorkspace()
{
    return;
}

Mki::Tensor torchTensor2AsdTensor(const torch::Tensor &torchTensor)
{
    Mki::Tensor asdTensor;
    asdTensor.data = torchTensor.data_ptr();
    asdTensor.dataSize = torchTensor.numel() * torchTensor.element_size();
    asdTensor.desc.dims.resize(torchTensor.sizes().size());

    for (uint64_t i = 0; i < torchTensor.sizes().size(); i++) {
        asdTensor.desc.dims[i] = torchTensor.sizes()[i];
    }

    asdTensor.desc.format = TENSOR_FORMAT_ND;

    auto it = DTYPE_MAP.find(torchTensor.scalar_type());
    if (it != DTYPE_MAP.end()) {
        asdTensor.desc.dtype = it->second;
    } else {
        ASDSIP_LOG(ERROR) << "not support dtype:" << torchTensor.scalar_type();
    }

    return asdTensor;
}


aclTensor* torchTensor2aclTensor(const torch::Tensor &torchTensor)
{
    Mki::Tensor asdTensor;
    asdTensor.data = torchTensor.data_ptr();
    asdTensor.dataSize = torchTensor.numel() * torchTensor.element_size();
    asdTensor.desc.dims.resize(torchTensor.sizes().size());

    for (uint64_t i = 0; i < torchTensor.sizes().size(); i++) {
        asdTensor.desc.dims[i] = torchTensor.sizes()[i];
    }

    asdTensor.desc.format = TENSOR_FORMAT_ND;

    auto it = DTYPE_MAP.find(torchTensor.scalar_type());
    if (it != DTYPE_MAP.end()) {
        asdTensor.desc.dtype = it->second;
    } else {
        ASDSIP_LOG(ERROR) << "not support dtype:" << torchTensor.scalar_type();
    }

    aclTensor *ret = nullptr;
    toAclTensor(asdTensor, ret);

    return ret;
}


void _setDevId(int32_t devId)
{
    global_devId = devId;
}

void _setDevice()
{
    MkiRtDeviceSetCurrent(global_devId);
    // _setGlobalWorkspace();
}

void _resetDevice()
{
    // _resetGlobalWorkspace();
    MkiRtDeviceResetCurrent(global_devId);
}

// uint8_t _transpose(const torch::Tensor &inTensor, int dim0, int dim1, torch::Tensor &outTensor)
// {
//     Mki::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = transpose(inAsdTensor, outAsdTensor, dim0, dim1, stream);
//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

uint8_t _swapLast2Axes(const torch::Tensor &inTensor, torch::Tensor &outTensor)
{
    aclTensor *inAsdTensor = torchTensor2aclTensor(inTensor);
    aclTensor *outAsdTensor = torchTensor2aclTensor(outTensor);

    MkiRtStream stream = getCurrentStream();

    AspbStatus status = swapLast2Axes(inAsdTensor, outAsdTensor, stream);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Something error in swapLast2Axes.";
        return -1;
    }

    return 0;
}

uint64_t _swapLast2AxesGetWorkspaceSize(const torch::Tensor &inTensor, torch::Tensor &outTensor)
{
    aclTensor *inAsdTensor = torchTensor2aclTensor(inTensor);
    aclTensor *outAsdTensor = torchTensor2aclTensor(outTensor);

    size_t work_size;
    AspbStatus status = swapLast2AxesGetWorkspaceSize(work_size);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Estimate error.";
    }

    return work_size;
}

aclTensor* torchF16Tensor2aclC32Tensor(const torch::Tensor &torchTensor)
{    
    if(torchTensor.numel() % 2 != 0){
        ASDSIP_LOG(ERROR) << "Total number of elements must be even (real+imag)";
    }

    int tensorDims = torchTensor.dim();
    auto tensorShape = torchTensor.sizes();

    Mki::SVector<int64_t> mkiShape;

    for (int i = 0; i < tensorDims - 1; i++) {
        mkiShape.push_back(static_cast<int64_t>(tensorShape[i]));
    }
    mkiShape.push_back(static_cast<int64_t>(tensorShape[tensorDims-1] / 2));

    Mki::Tensor asdTensor;

    asdTensor.data = torchTensor.data_ptr();
    asdTensor.dataSize = torchTensor.numel() * torchTensor.element_size();
    asdTensor.desc.dims.resize(tensorDims);
    asdTensor.desc.dims = mkiShape;

    asdTensor.desc.dtype = Mki::TENSOR_DTYPE_COMPLEX32;
    asdTensor.desc.format = TENSOR_FORMAT_ND;

    aclTensor *ret = nullptr;
    toAclTensor(asdTensor, ret);
    return ret;
}

uint8_t _asdMul(const int64_t n, torch::Tensor &x, torch::Tensor &y, torch::Tensor &result)
{
    aclTensor *xAsdTensor, *yAsdTensor, *resultAsdTensor;
    if (x.scalar_type() == torch::ScalarType::Half) {
        xAsdTensor = torchF16Tensor2aclC32Tensor(x);
        yAsdTensor = torchF16Tensor2aclC32Tensor(y);
        resultAsdTensor = torchF16Tensor2aclC32Tensor(result);
    } else {
        xAsdTensor = torchTensor2aclTensor(x);
        yAsdTensor = torchTensor2aclTensor(y);
        resultAsdTensor = torchTensor2aclTensor(result);
    }

    MkiRtStream stream = getCurrentStream();

    AspbStatus status = asdMul(n, xAsdTensor, yAsdTensor, resultAsdTensor, stream);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdMul error.";
        return -1;
    }
    return 0;
}

// uint8_t _conj(const torch::Tensor &inTensor, torch::Tensor &outTensor)
// {
//     Mki::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = Conj(inAsdTensor, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _sub(const torch::Tensor &inATensor, const torch::Tensor &inBTensor, torch::Tensor &outTensor)
// {
//     Mki::Tensor inAsdTensorA = torchTensor2AsdTensor(inATensor);
//     Mki::Tensor inAsdTensorB = torchTensor2AsdTensor(inBTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = sub(inAsdTensorA, inAsdTensorB, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _add(const torch::Tensor &inATensor, const torch::Tensor &inBTensor, torch::Tensor &outTensor)
// {
//     Mki::Tensor inAsdTensorA = torchTensor2AsdTensor(inATensor);
//     Mki::Tensor inAsdTensorB = torchTensor2AsdTensor(inBTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = add(inAsdTensorA, inAsdTensorB, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _matMul(const torch::Tensor &inATensor, const torch::Tensor &inBTensor,
//     int m, int k, int n, torch::Tensor &outTensor)
// {
//     Mki::Tensor inAsdTensorA = torchTensor2AsdTensor(inATensor);
//     Mki::Tensor inAsdTensorB = torchTensor2AsdTensor(inBTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = MatMul(inAsdTensorA, inAsdTensorB, outAsdTensor, m, k, n, stream);
//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _muls(const torch::Tensor &inTensor, const float scalar, torch::Tensor &outTensor)
// {
//     Mki::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = muls(inAsdTensor, scalar, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     MkiRtStreamSynchronize(stream);
//     CopyOutTensorToHost(outAsdTensor);
//     FreeTensorInDevice(inAsdTensor);
//     FreeTensorInDevice(outAsdTensor);

//     return 0;
// }

// uint8_t _cast(const torch::Tensor &inTensor, torch::Tensor &outTensor)
// {
//     Mki::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();
//     MallocTensorInDevice(inAsdTensor);
//     auto it = DTYPE_MAP.find(outTensor.scalar_type());
//     AspbStatus status = cast(inAsdTensor, outAsdTensor, stream, it->second);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//     }

//     return 0;
// }

// uint8_t _mul(const torch::Tensor &inATensor, const torch::Tensor &inBTensor, torch::Tensor &outTensor)
// {
//     Mki::Tensor inAsdTensorA = torchTensor2AsdTensor(inATensor);
//     Mki::Tensor inAsdTensorB = torchTensor2AsdTensor(inBTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = mul(inAsdTensorA, inAsdTensorB, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _zerolike(const torch::Tensor &inTensor, torch::Tensor &outTensor)
// {
//     Mki::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = zerosLike(inAsdTensor, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _complex_separation(const torch::Tensor &inTensor, torch::Tensor &realTensor, torch::Tensor &imgTensor)
// {
//     Mki::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     Mki::Tensor realAsdTensor = torchTensor2AsdTensor(realTensor);
//     Mki::Tensor imgAsdTensor = torchTensor2AsdTensor(imgTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = complex_separation(inAsdTensor, realAsdTensor, imgAsdTensor, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _complex_combination(const torch::Tensor &inRealTensor, torch::Tensor &inImgTensor, torch::Tensor &outTensor)
// {
//     Mki::Tensor inRealAsdTensor = torchTensor2AsdTensor(inRealTensor);
//     Mki::Tensor inImgAsdTensor = torchTensor2AsdTensor(inImgTensor);
//     Mki::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = complex_combination(inRealAsdTensor, inImgAsdTensor, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _ssyr2(const char uplo, const int64_t n,
//         const float alpha, torch::Tensor &x,
//         const int64_t incx, torch::Tensor &y,
//         const int64_t incy, torch::Tensor &A,
//         const int64_t lda)
// {
//     Mki::Tensor xAsdTensor = torchTensor2AsdTensor(x);
//     Mki::Tensor yAsdTensor = torchTensor2AsdTensor(y);
//     Mki::Tensor aAsdTensor = torchTensor2AsdTensor(A);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = ssyr2(uplo, n, alpha, xAsdTensor, incx, yAsdTensor, incy, aAsdTensor, lda, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _ssyr2v2(const char uplo, const int64_t n,
//         const float alpha, torch::Tensor &x,
//         const int64_t incx, torch::Tensor &y,
//         const int64_t incy, torch::Tensor &A,
//         const int64_t lda)
// {
//     Mki::Tensor xAsdTensor = torchTensor2AsdTensor(x);
//     Mki::Tensor yAsdTensor = torchTensor2AsdTensor(y);
//     Mki::Tensor aAsdTensor = torchTensor2AsdTensor(A);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = ssyr2v2(uplo, n, alpha, xAsdTensor, incx, yAsdTensor, incy, aAsdTensor, lda, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _ssyr2_v3(const char uplo, const int64_t n,
//         const float alpha, torch::Tensor &x,
//         const int64_t incx, torch::Tensor &y,
//         const int64_t incy, torch::Tensor &A,
//         const int64_t lda)
// {
//     Mki::Tensor xAsdTensor = torchTensor2AsdTensor(x);
//     Mki::Tensor yAsdTensor = torchTensor2AsdTensor(y);
//     Mki::Tensor aAsdTensor = torchTensor2AsdTensor(A);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = ssyr2_v3(uplo, n, alpha, xAsdTensor, incx, yAsdTensor, incy, aAsdTensor, lda, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _strmv(const char uplo, const char trans, const char diag,
//         const int64_t n, torch::Tensor &A, const int64_t lda,
//         torch::Tensor &x, const int64_t incx)
// {
//     Mki::Tensor xAsdTensor = torchTensor2AsdTensor(x);
//     Mki::Tensor aAsdTensor = torchTensor2AsdTensor(A);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = strmv(uplo, trans, diag, n, aAsdTensor, lda, xAsdTensor, incx, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _strmv_v2(const char uplo, const char trans, const char diag,
//         const int64_t n, torch::Tensor &A, const int64_t lda,
//         torch::Tensor &x, const int64_t incx)
// {
//     Mki::Tensor xAsdTensor = torchTensor2AsdTensor(x);
//     Mki::Tensor aAsdTensor = torchTensor2AsdTensor(A);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = strmv_v2(uplo, trans, diag, n, aAsdTensor, lda, xAsdTensor, incx, stream);
//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _strmm(const char side, const char uplo, const char trans, const char diag,
//              const int64_t m, const int64_t n, const float alpha, torch::Tensor &A,
//              const int64_t lda, torch::Tensor &B, const int64_t ldb,
//              torch::Tensor &C, const int64_t ldc)
// {
//     Mki::Tensor _A = torchTensor2AsdTensor(A);
//     Mki::Tensor _B = torchTensor2AsdTensor(B);
//     Mki::Tensor _C = torchTensor2AsdTensor(C);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = strmm(side, uplo, trans, diag, m, n, alpha, _A, lda, _B, ldb, _C, ldc, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _strmmV2(const char side, const char uplo, const char trans, const char diag,
//              const int64_t m, const int64_t n, const float alpha, torch::Tensor &A,
//              const int64_t lda, torch::Tensor &B, const int64_t ldb,
//              torch::Tensor &C, const int64_t ldc)
// {
//     Mki::Tensor _A = torchTensor2AsdTensor(A);
//     Mki::Tensor _B = torchTensor2AsdTensor(B);
//     Mki::Tensor _C = torchTensor2AsdTensor(C);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = strmmV2(side, uplo, trans, diag, m, n, alpha, _A, lda, _B, ldb, _C, ldc, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _strmm_v3(const char side, const char uplo, const char trans, const char diag,
//              const int64_t m, const int64_t n, const float alpha, torch::Tensor &A,
//              const int64_t lda, torch::Tensor &B, const int64_t ldb,
//              torch::Tensor &C, const int64_t ldc)
// {
//     Mki::Tensor _A = torchTensor2AsdTensor(A);
//     Mki::Tensor _B = torchTensor2AsdTensor(B);
//     Mki::Tensor _C = torchTensor2AsdTensor(C);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = strmm_v3(side, uplo, trans, diag, m, n, alpha, _A, lda, _B, ldb, _C, ldc, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }
//     return 0;
// }

// uint8_t _strsm(const char side, const char uplo, const char transa, const char diag, const int64_t m,
//                 const int64_t n, const float alpha, torch::Tensor &A,  const int64_t lda,
//                 torch::Tensor &B, const int64_t ldb)
// {
//     Mki::Tensor _A = torchTensor2AsdTensor(A);
//     Mki::Tensor _B = torchTensor2AsdTensor(B);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = strsm(side, uplo, transa, diag, m, n, alpha, _A, lda, _B, ldb, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _ctrmv(const char uplo, const char trans,
//         const char diag, const int64_t n,
//         torch::Tensor &A, const int64_t lda,
//         torch::Tensor &x, const int64_t incx)
// {
//     Mki::Tensor xAsdTensor = torchTensor2AsdTensor(x);
//     Mki::Tensor aAsdTensor = torchTensor2AsdTensor(A);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = ctrmv(uplo, trans, diag, n, aAsdTensor, lda, xAsdTensor, incx, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _ctrmv_v2(const char uplo, const char trans,
//         const char diag, const int64_t n,
//         torch::Tensor &A, const int64_t lda,
//         torch::Tensor &x, const int64_t incx)
// {
//     Mki::Tensor xAsdTensor = torchTensor2AsdTensor(x);
//     Mki::Tensor aAsdTensor = torchTensor2AsdTensor(A);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = ctrmv_v2(uplo, trans, diag, n, aAsdTensor, lda, xAsdTensor, incx, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _cgemm(const char transA, const char transB, const int64_t m, const int64_t n,
//         const int64_t k, const std::complex<float> alpha, torch::Tensor &A,
//         const int64_t lda, torch::Tensor &B, const int64_t ldb, const std::complex<float> beta,
//         torch::Tensor &C, const int64_t ldc)
// {
//     Mki::Tensor asdTensorA = torchTensor2AsdTensor(A);
//     Mki::Tensor asdTensorB = torchTensor2AsdTensor(B);
//     Mki::Tensor asdTensorC = torchTensor2AsdTensor(C);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = cgemm(
//         transA, transB, m, n, k, alpha, asdTensorA, lda, asdTensorB, ldb, beta, asdTensorC, ldc, stream);

//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _sgetrf(const int64_t n, torch::Tensor &A, const int64_t lda, torch::Tensor &pivot, torch::Tensor &info)
// {
//     Mki::Tensor asdTensorA = torchTensor2AsdTensor(A);
//     Mki::Tensor asdPivotTensor = torchTensor2AsdTensor(pivot);
//     Mki::Tensor asdInfoTensor = torchTensor2AsdTensor(info);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = sgetrf(n, asdTensorA, lda, asdPivotTensor, asdInfoTensor, stream);
//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _sgetri(const int64_t n, torch::Tensor &A, const int64_t lda, torch::Tensor &pivot,
//                     torch::Tensor &C, const int64_t ldc, torch::Tensor &info)
// {
//     Mki::Tensor asdTensorA = torchTensor2AsdTensor(A);
//     Mki::Tensor asdPivotTensor = torchTensor2AsdTensor(pivot);
//     Mki::Tensor asdInfoTensor = torchTensor2AsdTensor(info);
//     Mki::Tensor asdTensorC = torchTensor2AsdTensor(C);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = sgetri(n, asdTensorA, n, asdPivotTensor, asdTensorC, ldc, 0, stream);
//     if (!status.Ok()) {
//         ASDSIP_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

uint8_t _asdBlasMakeSwapPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeSwapPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeSwapPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasSswap(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &y,
                      const int64_t incy)
{
    aclTensor* xAsdTensor = torchTensor2aclTensor(x);
    aclTensor* yAsdTensor = torchTensor2aclTensor(y);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasSswap(_handle, n, xAsdTensor, incx, yAsdTensor, incy);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasSswap error.";
        return -1;
    }
    return 0;
}
uint8_t _asdBlasCswap(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &y,
                      const int64_t incy)
{
    aclTensor* xAsdTensor = torchTensor2aclTensor(x);
    aclTensor* yAsdTensor = torchTensor2aclTensor(y);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCswap(_handle, n, xAsdTensor, incx, yAsdTensor, incy);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCswap error.";
        return -1;
    }
    return 0;
}

uint64_t _fftInit(const int64_t fft_size, const py::object in_dtype, const py::object out_dtype,
                  const int64_t batch_size, const bool forward)
{
    auto inIt = DTYPE_MAP.find(torch::python::detail::py_object_to_dtype(in_dtype));
    auto outIt = DTYPE_MAP.find(torch::python::detail::py_object_to_dtype(out_dtype));
    auto in_AsdDtype = inIt->second;
    auto out_AsdDtype = outIt->second;
    MkiRtStream stream = getCurrentStream();

    asdFftHandle handle;
    asdFftType fft_type;
    if (in_AsdDtype == TENSOR_DTYPE_COMPLEX64 && out_AsdDtype == TENSOR_DTYPE_COMPLEX64) {
        fft_type = asdFftType::ASCEND_FFT_C2C;
    } else if (in_AsdDtype == TENSOR_DTYPE_FLOAT && out_AsdDtype == TENSOR_DTYPE_COMPLEX64) {
        fft_type = asdFftType::ASCEND_FFT_R2C;
    } else if (in_AsdDtype == TENSOR_DTYPE_COMPLEX64 && out_AsdDtype == TENSOR_DTYPE_FLOAT) {
        fft_type = asdFftType::ASCEND_FFT_C2R;
    } else {
        ASDSIP_LOG(ERROR) << "asdFftType error.";
    }
    asdFftCreate(handle);
    asdFftMakePlan1D(handle, fft_size, fft_type,
                     forward ? asdFftDirection::ASCEND_FFT_FORWARD : asdFftDirection::ASCEND_FFT_INVERSE, batch_size);
    asdFftSetStream(handle, stream);
    return (uint64_t)handle;
}

uint64_t _fftInit1DCol(const int64_t fft_size, const py::object in_dtype, const py::object out_dtype,
                       const int64_t batch_size, const bool forward)
{
    auto inIt = DTYPE_MAP.find(torch::python::detail::py_object_to_dtype(in_dtype));
    auto outIt = DTYPE_MAP.find(torch::python::detail::py_object_to_dtype(out_dtype));
    auto in_AsdDtype = inIt->second;
    auto out_AsdDtype = outIt->second;
    MkiRtStream stream = getCurrentStream();

    asdFftHandle handle;
    asdFftType fft_type;
    if (in_AsdDtype == TENSOR_DTYPE_COMPLEX64 && out_AsdDtype == TENSOR_DTYPE_COMPLEX64) {
        fft_type = asdFftType::ASCEND_FFT_C2C;
    } else {
        ASDSIP_LOG(ERROR) << "asdFftType error.";
    }
    asdFftCreate(handle);
    asdFftMakePlan1D(handle, fft_size, fft_type,
                     forward ? asdFftDirection::ASCEND_FFT_FORWARD : asdFftDirection::ASCEND_FFT_INVERSE, batch_size,
                     asdFft1dDimType::ASCEND_FFT_VERTICAL);
    asdFftSetStream(handle, stream);
    return (uint64_t)handle;
}

uint64_t _fftInit2D(const int64_t fft_size_x, const int64_t fft_size_y, const py::object in_dtype,
                    const py::object out_dtype, const int64_t batch_size, const bool forward)
{
    auto inIt = DTYPE_MAP.find(torch::python::detail::py_object_to_dtype(in_dtype));
    auto outIt = DTYPE_MAP.find(torch::python::detail::py_object_to_dtype(out_dtype));
    auto in_AsdDtype = inIt->second;
    auto out_AsdDtype = outIt->second;
    MkiRtStream stream = getCurrentStream();

    asdFftHandle handle;
    asdFftType fft_type;
    if (in_AsdDtype == TENSOR_DTYPE_COMPLEX64 && out_AsdDtype == TENSOR_DTYPE_COMPLEX64) {
        fft_type = asdFftType::ASCEND_FFT_C2C;
    } else if (in_AsdDtype == TENSOR_DTYPE_FLOAT && out_AsdDtype == TENSOR_DTYPE_COMPLEX64) {
        fft_type = asdFftType::ASCEND_FFT_R2C;
    } else if (in_AsdDtype == TENSOR_DTYPE_COMPLEX64 && out_AsdDtype == TENSOR_DTYPE_FLOAT) {
        fft_type = asdFftType::ASCEND_FFT_C2R;
    } else {
        ASDSIP_LOG(ERROR) << "asdFftType error.";
    }
    asdFftCreate(handle);
    asdFftMakePlan2D(handle, fft_size_x, fft_size_y, fft_type,
                     forward ? asdFftDirection::ASCEND_FFT_FORWARD : asdFftDirection::ASCEND_FFT_INVERSE, batch_size);
    asdFftSetStream(handle, stream);
    return (uint64_t)handle;
}

uint64_t _fftGetWorkspaceSize(uint64_t handle)
{
    asdFftHandle handle_ = (void *)handle;

    size_t work_size;
    AspbStatus status = asdFftGetWorkspaceSize(handle_, work_size);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Estimate error.";
    }

    return work_size;
}

uint8_t _fftSetWorkspace(uint64_t handle, torch::Tensor workspace)
{
    asdFftHandle handle_ = (void *)handle;

    Mki::Tensor workspaceTensor = torchTensor2AsdTensor(workspace);

    AspbStatus status = asdFftSetWorkspace(handle_, (uint8_t *)workspaceTensor.data);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "SetWorkspace error.";
    }

    return 0;
}

uint8_t _fftExecute(uint64_t handle, torch::Tensor in, torch::Tensor out)
{
    asdFftHandle handle_ = (void *)handle;
    aclTensor* inAsdTensor = torchTensor2aclTensor(in);
    aclTensor* outAsdTensor = torchTensor2aclTensor(out);

    asdFftType fft_type;
    AspbStatus status = asdFftGetType(handle_, fft_type);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "GetFFTType error.";
    }

    if (fft_type == asdFftType::ASCEND_FFT_C2C) {
        status = asdFftExecC2C(handle_, inAsdTensor, outAsdTensor);
    } else if (fft_type == asdFftType::ASCEND_FFT_R2C) {
        status = asdFftExecR2C(handle_, inAsdTensor, outAsdTensor);
    } else if (fft_type == asdFftType::ASCEND_FFT_C2R) {
        status = asdFftExecC2R(handle_, inAsdTensor, outAsdTensor);
    } else {
        ASDSIP_LOG(ERROR) << "asdFftType error.";
    }

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Exec error.";
    }

    return 0;
}

uint8_t _fftFinalize(uint64_t handle)
{
    asdFftHandle handle_ = (void *)handle;
    asdFftDestroy(handle_);
    return 0;
}

// uint8_t _asStrided(const torch::Tensor &inTensor, std::list<int64_t> size,
//     std::list<int64_t> stride, std::list<int64_t> offset, torch::Tensor &outTensor)
// {
//     AsdOps::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     AsdOps::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     AsdOps::SVector<int64_t> vectorSize;
//     for (auto value : size) {
//         vectorSize.push_back(value);
//     }

//     AsdOps::SVector<int64_t> vectorStride;
//     for (auto value : stride) {
//         vectorStride.push_back(value);
//     }

//     AsdOps::SVector<int64_t> vectorOffset;
//     for (auto value : offset) {
//         vectorOffset.push_back(value);
//     }

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = AsStrided(inAsdTensor, outAsdTensor, vectorSize, vectorStride, stream, vectorOffset);

//     if (!status.Ok()) {
//         ASD_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _modulus(const torch::Tensor &inTensor, torch::Tensor &outTensor)
// {
//     AsdOps::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     AsdOps::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = modulus_v2(inAsdTensor, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASD_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _PostCFARDetection(const torch::Tensor &inTensor,
//                             const int64_t trainLength,
//                             const int64_t guardLength,
//                             const float coef,
//                             const int64_t trainType,
//                             const int64_t isTrans,
//                             torch::Tensor &outTensor)
// {
//     AsdOps::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     AsdOps::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = cfarDetection(inAsdTensor, trainLength, guardLength, coef, trainType, isTrans, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASD_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _padZero(const torch::Tensor &inTensor, torch::Tensor &outTensor)
// {
//     AsdOps::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     AsdOps::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = padZero(inAsdTensor, outAsdTensor, stream);

//     if (!status.Ok()) {
//         ASD_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

// uint8_t _cutZero(const torch::Tensor &inTensor, torch::Tensor &outTensor, const uint32_t dstShape0Size)
// {
//     AsdOps::Tensor inAsdTensor = torchTensor2AsdTensor(inTensor);
//     AsdOps::Tensor outAsdTensor = torchTensor2AsdTensor(outTensor);

//     MkiRtStream stream = getCurrentStream();

//     AspbStatus status = cutZero(inAsdTensor, outAsdTensor, dstShape0Size, stream);

//     if (!status.Ok()) {
//         ASD_LOG(ERROR) << "Something error.";
//         return -1;
//     }

//     return 0;
// }

uint64_t _setBlasWorkspace(uint64_t bufferSize = 256 * 1024 * 1024)
{
    uint8_t *blasBuffer = nullptr;
    MkiRtMemMallocDevice((void **)&blasBuffer, bufferSize, MKIRT_MEM_DEFAULT);
    return (uint64_t)blasBuffer;
}

uint64_t _asdBlasCreateHandle()
{
    asdBlasHandle handle;
    AspbStatus status = asdBlasCreate(handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "asdBlasCreateHandle error.";
        return -1;
    }
    return (uint64_t)handle;
}

uint8_t _asdBlasMakeSsyr2Plan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeSsyr2Plan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeSsyr2Plan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeCgemmPlan(uint64_t handle, const char transa, const char transb, int64_t M, int64_t N, int64_t K,
                              int64_t lda, int64_t ldb, int64_t ldc)
{
    asdBlasHandle _handle = (void *)handle;

    auto transaValue = OPERATION_MAP.find(transa);
    if (transaValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support transa value:" << transa;
    }
    auto transbValue = OPERATION_MAP.find(transb);
    if (transbValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support transb value:" << transb;
    }

    AspbStatus status = asdBlasMakeCgemmPlan(_handle, transaValue->second, transbValue->second, M, N, K, lda, ldb, ldc);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeCgemmPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeHCgemmBatchedPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasMakeHCgemmBatchedPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeHCgemmBatchedPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeCgemmBatchedPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasMakeCgemmBatchedPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeCgemmBatchedPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeAsumPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeAsumPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeAsumPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeNrm2Plan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeNrm2Plan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeNrm2Plan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeCopyPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeCopyPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeCopyPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeCalPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeCalPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeCalPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeDotPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeDotPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeDotPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeRotPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeRotPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeRotPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeSsyrPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeSsyrPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeRotPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeStrmmPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeStrmmPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeStrmmPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeColwiseMulPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeColwiseMulPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeColwiseMulPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeComplexMatDotPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeComplexMatDotPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeComplexMatDotPlan error.";
        return -1;
    }
    return 0;
}

// uint8_t _asdBlasMakeSmatinvBatchedPlan(uint64_t handle, int64_t n, uint64_t workspace) {

//     asdBlasHandle _handle = (void *)handle;
//     uint8_t *workspaceBuffer = (uint8_t *)workspace;
//     AspbStatus status = asdBlasMakeSmatinvBatchedPlan(_handle, n, workspaceBuffer);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasMakeSmatinvBatchedPlan error.";
//         return -1;
//     }
//     return 0;
// }

// uint8_t _asdBlasMakeCmatinvBatchedPlan(uint64_t handle, int64_t n, uint64_t workspace) {

//     asdBlasHandle _handle = (void *)handle;
//     uint8_t *workspaceBuffer = (uint8_t *)workspace;
//     AspbStatus status = asdBlasMakeCmatinvBatchedPlan(_handle, n, workspaceBuffer);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "asdBlasMakeCmatinvBatchedPlan error.";
//         return -1;
//     }
//     return 0;
// }

uint8_t _asdBlasMakeIamaxPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeIamaxPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeIamaxPlan error.";
        return -1;
    }
    return 0;
}

// uint8_t _asdBlasMakeIaminPlan(uint64_t handle)
// {
//     asdBlasHandle _handle = (void *)handle;
//     AspbStatus status = asdBlasMakeIaminPlan(_handle);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasMakeIaminPlan error.";
//         return -1;
//     }
//     return 0;
// }

uint8_t _asdBlasMakeCtrmvPlan(uint64_t handle, const char uplo, int64_t n)
{
    asdBlasHandle _handle = (void *)handle;

    auto uploValue = FILL_MODE_MAP.find(uplo);
    if (uploValue == FILL_MODE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
    }

    AspbStatus status = asdBlasMakeCtrmvPlan(_handle, uploValue->second, n);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeCtrmvPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasSasum(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &result)
{
    aclTensor* xAsdTensor = torchTensor2aclTensor(x);
    aclTensor* resultAsdTensor = torchTensor2aclTensor(result);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasSasum(_handle, n, xAsdTensor, incx, resultAsdTensor);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasSasum error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasScasum(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &result)
{
    aclTensor* xAsdTensor = torchTensor2aclTensor(x);
    aclTensor* resultAsdTensor = torchTensor2aclTensor(result);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasScasum(_handle, n, xAsdTensor, incx, resultAsdTensor);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasScasum error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasSnrm2(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &result)
{
    aclTensor* xAsdTensor = torchTensor2aclTensor(x);
    aclTensor* resultAsdTensor = torchTensor2aclTensor(result);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasSnrm2(_handle, n, xAsdTensor, incx, resultAsdTensor);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasSnrm2 error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasScnrm2(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &result)
{
    aclTensor* xAsdTensor = torchTensor2aclTensor(x);
    aclTensor* resultAsdTensor = torchTensor2aclTensor(result);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasScnrm2(_handle, n, xAsdTensor, incx, resultAsdTensor);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasScnrm2 error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasScopy(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &y,
                      const int64_t incy)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasScopy(_handle, n, xAsdTensor, incx, yAsdTensor, incy);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasScopy error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCcopy(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &y,
                      const int64_t incy)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCcopy(_handle, n, xAsdTensor, incx, yAsdTensor, incy);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCcopy error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasSscal(uint64_t &handle, const int64_t n, const float alpha, torch::Tensor &x, const int64_t incx)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasSscal(_handle, n, alpha, xAsdTensor, incx);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasSscal error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCscal(uint64_t &handle, const int64_t n, const std::complex<float> alpha, torch::Tensor &x,
                      const int64_t incx)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCscal(_handle, n, alpha, xAsdTensor, incx);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCscal error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCsscal(uint64_t &handle, const int64_t n, const float alpha, torch::Tensor &x, const int64_t incx)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCsscal(_handle, n, alpha, xAsdTensor, incx);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCsscal error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCtrmv(uint64_t &handle, const char uplo, const char trans, const char diag, const int64_t n,
                      torch::Tensor &A, const int64_t lda, torch::Tensor &x, const int64_t incx)
{
    auto uploValue = FILL_MODE_MAP.find(uplo);
    if (uploValue == FILL_MODE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
    }
    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
    }
    auto diagValue = DIAG_TYPE_MAP.find(diag);
    if (diagValue == DIAG_TYPE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support diag value:" << diag;
    }

    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *aAsdTensor = torchTensor2aclTensor(A);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCtrmv(_handle, uploValue->second, transValue->second, diagValue->second, n, aAsdTensor,
                                     lda, xAsdTensor, incx);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCtrmv error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasSetStream(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    MkiRtStream stream = getCurrentStream();
    AspbStatus status = asdBlasSetStream(_handle, stream);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasSetStream error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasSsyr2(uint64_t handle, const char uplo, const int64_t n, const float alpha, const torch::Tensor &x,
                      const int64_t incx, const torch::Tensor &y, const int64_t incy, const torch::Tensor &A,
                      const int64_t lda)
{
    auto uploValue = FILL_MODE_MAP.find(uplo);
    if (uploValue == FILL_MODE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
    }

    asdBlasHandle _handle = (void *)handle;

    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    aclTensor *aAsdTensor = torchTensor2aclTensor(A);

    AspbStatus status = asdBlasSsyr2(_handle, uploValue->second, n, alpha, xAsdTensor, incx, yAsdTensor, incy,
                                     aAsdTensor, lda);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasSsyr2 error.";
        return -1;
    }

    return 0;
}

uint8_t _asdBlasSsyr(uint64_t handle, const char uplo, const int64_t n, const float alpha, const torch::Tensor &x,
                     const int64_t incx, const torch::Tensor &A, const int64_t lda)
{
    auto uploValue = FILL_MODE_MAP.find(uplo);
    if (uploValue == FILL_MODE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
    }

    asdBlasHandle _handle = (void *)handle;

    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *aAsdTensor = torchTensor2aclTensor(A);

    AspbStatus status = asdBlasSsyr(_handle, uploValue->second, n, alpha, xAsdTensor, incx, aAsdTensor, lda);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasSsyr error.";
        return -1;
    }

    return 0;
}

uint8_t _asdBlasStrmm(uint64_t handle, const char side, const char uplo, const char trans, const char diag,
                      const int64_t m, const int64_t n, const float alpha, torch::Tensor &A, const int64_t lda,
                      torch::Tensor &B, const int64_t ldb, torch::Tensor &C, const int64_t ldc)
{
    auto uploValue = FILL_MODE_MAP.find(uplo);
    if (uploValue == FILL_MODE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
    }
    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support transa value:" << trans;
    }
    auto diagValue = DIAG_TYPE_MAP.find(diag);
    if (diagValue == DIAG_TYPE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support diag value:" << diag;
    }
    auto sideValue = SIDE_MODE_MAP.find(side);
    if (sideValue == SIDE_MODE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support side value:" << side;
    }

    asdBlasHandle _handle = (void *)handle;

    aclTensor *asdTensorA = torchTensor2aclTensor(A);
    aclTensor *asdTensorB = torchTensor2aclTensor(B);
    aclTensor *asdTensorC = torchTensor2aclTensor(C);

    AspbStatus status = asdBlasStrmm(_handle, sideValue->second, uploValue->second, transValue->second,
                                     diagValue->second, m, n, alpha, asdTensorA, lda, asdTensorB, ldb, asdTensorC, ldc);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasStrmm error.";
        return -1;
    }

    return 0;
}

// uint8_t _asdBlasMakeCsyrkPlan(uint64_t &handle, const char uplo, const int64_t n, const int64_t k) {
//     asdBlasHandle _handle = (void *)handle;
//     auto uploValue = FILL_MODE_MAP.find(uplo);
//     if (uploValue == FILL_MODE_MAP.end()) {
//         ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
//     }

//     AspbStatus status = asdBlasMakeCsyrkPlan(_handle, uploValue->second, n, k);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasMakeCsyrkPlan error.";
//         return -1;
//     }
//     return 0;
// }

// uint8_t _asdBlasCsyrk(uint64_t &handle, const char uplo, const char trans, const int64_t n,
//                         const int64_t k, const std::complex<float>* alpha, const torch::Tensor &A, const int64_t lda,
//                         const std::complex<float>* beta, torch::Tensor &C, const int64_t ldc)
// {
//     auto uploValue = FILL_MODE_MAP.find(uplo);
//     if (uploValue == FILL_MODE_MAP.end()) {
//         ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
//     }
//     auto transValue = OPERATION_MAP.find(trans);
//     if (transValue == OPERATION_MAP.end()) {
//         ASDSIP_LOG(ERROR) << "not support transa value:" << trans;
//     }

//     asdBlasHandle _handle = (void *)handle;
//     Mki::Tensor asdTensorA = torchTensor2AsdTensor(A);
//     Mki::Tensor asdTensorC = torchTensor2AsdTensor(C);

//     AspbStatus status = asdBlasCsyrk(_handle, uploValue->second, transValue->second, n, k, alpha, asdTensorA, n, beta, asdTensorC, n);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasCsyrk error.";
//         return -1;
//     }

//     return 0;
// }

uint8_t _asdBlasCsrot(uint64_t handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &y,
                      const int64_t incy, const float c, const float s)
{
    asdBlasHandle _handle = (void *)handle;

    aclTensor *asdTensorX = torchTensor2aclTensor(x);
    aclTensor *asdTensorY = torchTensor2aclTensor(y);

    AspbStatus status = asdBlasCsrot(_handle, n, asdTensorX, incx, asdTensorY, incy, c, s);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCsrot error.";
        return -1;
    }

    return 0;
}

uint8_t _asdBlasCgerc(uint64_t &handle, const int64_t m, const int64_t n, const std::complex<float> alpha,
                      torch::Tensor &x, const int64_t incx, torch::Tensor &y, const int64_t incy, torch::Tensor &A,
                      const int64_t lda)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    aclTensor *aAsdTensor = torchTensor2aclTensor(A);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCgerc(_handle, m, n, alpha, xAsdTensor, incx, yAsdTensor, incy, aAsdTensor, lda);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCgerc error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCgemm(uint64_t handle, const char transA, const char transB, const int64_t m, const int64_t n,
                      const int64_t k, const std::complex<float> alpha, torch::Tensor &A, const int64_t lda,
                      torch::Tensor &B, const int64_t ldb, const std::complex<float> beta, torch::Tensor &C,
                      const int64_t ldc)
{
    asdBlasHandle _handle = (void *)handle;
    
    auto transAValue = OPERATION_MAP.find(transA);
    if (transAValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support transA value:" << transA;
    }
    auto transBValue = OPERATION_MAP.find(transB);
    if (transBValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support transB value:" << transB;
    }

    aclTensor *asdTensorA = torchTensor2aclTensor(A);
    aclTensor *asdTensorB = torchTensor2aclTensor(B);
    aclTensor *asdTensorC = torchTensor2aclTensor(C);

    AspbStatus status = asdBlasCgemm(_handle, transAValue->second, transBValue->second, m, n, k, alpha, asdTensorA,
                                     lda, asdTensorB, ldb, beta, asdTensorC, ldc);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCgemm error.";
        return -1;
    }

    return 0;
}

uint8_t _asdBlasHCgemmBatched(uint64_t handle, const char transA, const char transB, const int64_t m, const int64_t n,
                              const int64_t k, const std::complex<op::fp16_t> alpha, torch::Tensor &A, 
                              const int64_t lda, torch::Tensor &B, const int64_t ldb, 
                              const std::complex<op::fp16_t> beta, torch::Tensor &C, const int64_t ldc, 
                              int64_t batchCount)
{
    asdBlasHandle _handle = (void *)handle;

    auto transAValue = OPERATION_MAP.find(transA);
    if (transAValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support transA value:" << transA;
    }
    auto transBValue = OPERATION_MAP.find(transB);
    if (transBValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support transB value:" << transB;
    }

    aclTensor *asdTensorA = torchTensor2aclTensor(A);
    aclTensor *asdTensorB = torchTensor2aclTensor(B);
    aclTensor *asdTensorC = torchTensor2aclTensor(C);

    AspbStatus status = asdBlasHCgemmBatched(
        _handle, transAValue->second, transBValue->second, m, n, k, alpha, asdTensorA,
        lda, asdTensorB, ldb, beta, asdTensorC, ldc, batchCount);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasHCgemmBatched error.";
        return -1;
    }

    return 0;
}

uint8_t _asdBlasCgemmBatched(uint64_t handle, const char transA, const char transB, const int64_t m, const int64_t n,
                              const int64_t k, const std::complex<float> alpha, torch::Tensor &A, 
                              const int64_t lda, torch::Tensor &B, const int64_t ldb, 
                              const std::complex<float> beta, torch::Tensor &C, const int64_t ldc, 
                              int64_t batchCount)
{
    asdBlasHandle _handle = (void *)handle;

    auto transAValue = OPERATION_MAP.find(transA);
    if (transAValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support transA value:" << transA;
    }
    auto transBValue = OPERATION_MAP.find(transB);
    if (transBValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support transB value:" << transB;
    }

    aclTensor *asdTensorA = torchTensor2aclTensor(A);
    aclTensor *asdTensorB = torchTensor2aclTensor(B);
    aclTensor *asdTensorC = torchTensor2aclTensor(C);

    AspbStatus status = asdBlasCgemmBatched(
        _handle, transAValue->second, transBValue->second, m, n, k, alpha, asdTensorA,
        lda, asdTensorB, ldb, beta, asdTensorC, ldc, batchCount);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCgemmBatched error.";
        return -1;
    }

    return 0;
}


uint8_t _asdMakeBlasStrmvPlan(uint64_t handle, const char uplo, const char trans, int64_t n)
{
    asdBlasHandle _handle = (void *)handle;

    auto uploValue = FILL_MODE_MAP.find(uplo);
    if (uploValue == FILL_MODE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
    }
    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
    }

    AspbStatus status = asdBlasMakeStrmvPlan(_handle, uploValue->second, transValue->second, n);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeStrmvPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasStrmv(uint64_t handle, const char uplo, const char trans, const char diag, const int64_t n,
                      torch::Tensor &A, const int64_t lda, torch::Tensor &x, const int64_t incx)
{
    auto uploValue = FILL_MODE_MAP.find(uplo);
    if (uploValue == FILL_MODE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
    }
    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
    }
    auto diagValue = DIAG_TYPE_MAP.find(diag);
    if (diagValue == DIAG_TYPE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support diag value:" << diag;
    }

    aclTensor* aAsdTensor = torchTensor2aclTensor(A);
    aclTensor* xAsdTensor = torchTensor2aclTensor(x);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasStrmv(_handle, uploValue->second, transValue->second, diagValue->second, n, aAsdTensor,
                                     lda, xAsdTensor, incx);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasStrmv error.";
        return -1;
    }

    return 0;
}

uint8_t _asdBlasMakeCgercPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeCgercPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeCgercPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeCgemvPlan(uint64_t handle, const char trans, const int64_t m, const int64_t n, torch::Tensor &y,
                              const int64_t incy)
{
    asdBlasHandle _handle = (void *)handle;

    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
    }

    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    AspbStatus status = asdBlasMakeCgemvPlan(_handle, transValue->second, m, n, yAsdTensor, incy);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeCgemvPlan error.";
        return -1;
    }
    return 0;
}


uint8_t _asdBlasMakeHCgemvBatchedPlan(uint64_t handle, const char trans, const int64_t m)
{
    asdBlasHandle _handle = (void *)handle;

    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
    }

    AspbStatus status = asdBlasMakeHCgemvBatchedPlan(_handle, transValue->second, m);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeHCgemvBatchedPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasMakeCgemvBatchedPlan(uint64_t handle, const char trans, const int64_t m)
{
    asdBlasHandle _handle = (void *)handle;

    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
    }

    AspbStatus status = asdBlasMakeCgemvBatchedPlan(_handle, transValue->second, m);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeCgemvBatchedPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCgemv(uint64_t &handle, const char trans, const int64_t m, const int64_t n,
                      const std::complex<float> alpha, torch::Tensor &A, const int64_t lda, torch::Tensor &x,
                      const int64_t incx, const std::complex<float> beta, torch::Tensor &y, const int64_t incy)
{
    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
    }

    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    aclTensor *aAsdTensor = torchTensor2aclTensor(A);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status =
        asdBlasCgemv(_handle, transValue->second, m, n, alpha, aAsdTensor, lda, xAsdTensor, incx, beta, yAsdTensor, incy);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCgemv error.";
        return -1;
    }

    return 0;
}

uint8_t _asdBlasSdot(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &y,
                     const int64_t incy, torch::Tensor &result)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    aclTensor *aAsdTensor = torchTensor2aclTensor(result);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasSdot(_handle, n, xAsdTensor, incx, yAsdTensor, incy, aAsdTensor);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasSdot error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCdotu(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &y,
                      const int64_t incy, torch::Tensor &result)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    aclTensor *aAsdTensor = torchTensor2aclTensor(result);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCdotu(_handle, n, xAsdTensor, incx, yAsdTensor, incy, aAsdTensor);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCdotu error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCdotc(uint64_t &handle, const int64_t n, torch::Tensor &x, const int64_t incx, torch::Tensor &y,
                      const int64_t incy, torch::Tensor &result)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    aclTensor *aAsdTensor = torchTensor2aclTensor(result);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCdotc(_handle, n, xAsdTensor, incx, yAsdTensor, incy, aAsdTensor);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCdotc error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasColwiseMul(uint64_t &handle, const int64_t m, const int64_t n, torch::Tensor &mat, torch::Tensor &vec,
                           torch::Tensor &result)
{
    aclTensor *matAsdTensor = torchTensor2aclTensor(mat);
    aclTensor *vecAsdTensor = torchTensor2aclTensor(vec);
    aclTensor *resultAsdTensor = torchTensor2aclTensor(result);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasColwiseMul(_handle, m, n, matAsdTensor, vecAsdTensor, resultAsdTensor);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasColwiseMul error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasComplexMatDot(uint64_t &handle, const int64_t m, const int64_t n, torch::Tensor &matx,
                              torch::Tensor &maty, torch::Tensor &result)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(matx);
    aclTensor *yAsdTensor = torchTensor2aclTensor(maty);
    aclTensor *resultAsdTensor = torchTensor2aclTensor(result);

    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasComplexMatDot(_handle, m, n, xAsdTensor, yAsdTensor, resultAsdTensor);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasComplexMatDot error.";
        return -1;
    }
    return 0;
}

// uint8_t _asdBlasSmatinvBatched(uint64_t &handle, const int64_t n,
//                             torch::Tensor &A, const int64_t lda,
//                             torch::Tensor &Ainv, const int64_t lda_inv,
//                             torch::Tensor &info, const int64_t batchSize) {
//     Mki::Tensor matAsdTensor = torchTensor2AsdTensor(A);
//     Mki::Tensor matAinvsdTensor = torchTensor2AsdTensor(Ainv);
//     Mki::Tensor resultAsdTensor = torchTensor2AsdTensor(info);
//     asdBlasHandle _handle = (void *)handle;

//     AspbStatus status = asdBlasSmatinvBatched(_handle, n, matAsdTensor, lda,
//                                                 matAinvsdTensor, lda_inv,
//                                                 resultAsdTensor, batchSize);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasSmatinvBatched error.";
//         return -1;
//     }
//     return 0;
// }

// uint8_t _asdBlasCmatinvBatched(uint64_t &handle, const int64_t n,
//                             torch::Tensor &A, const int64_t lda,
//                             torch::Tensor &Ainv, const int64_t lda_inv,
//                             torch::Tensor &info, const int64_t batchSize) {
//     Mki::Tensor matAsdTensor = torchTensor2AsdTensor(A);
//     Mki::Tensor matAinvsdTensor = torchTensor2AsdTensor(Ainv);
//     Mki::Tensor resultAsdTensor = torchTensor2AsdTensor(info);
//     asdBlasHandle _handle = (void *)handle;

//     AspbStatus status = asdBlasCmatinvBatched(_handle, n, matAsdTensor, lda,
//                                                 matAinvsdTensor, lda_inv,
//                                                 resultAsdTensor, batchSize);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasCmatinvBatched error.";
//         return -1;
//     }
//     return 0;
// }

uint8_t _asdBlasIsamax(uint64_t &handle, const int64_t n, torch::Tensor &A, const int64_t incx, torch::Tensor &result)
{
    aclTensor *ATensor = torchTensor2aclTensor(A);
    aclTensor *resultTensor = torchTensor2aclTensor(result);

    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasIsamax(_handle, n, ATensor, incx, resultTensor);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasIsamax error.";
        return -1;
    }
    return 0;
}

// uint8_t _asdBlasIsamin(uint64_t &handle, const int64_t n, torch::Tensor &A, const int64_t incx, torch::Tensor &result)
// {
//     Mki::Tensor ATensor = torchTensor2AsdTensor(A);
//     Mki::Tensor resultTensor = torchTensor2AsdTensor(result);

//     asdBlasHandle _handle = (void *)handle;

//     AspbStatus status = asdBlasIsamin(_handle, n, ATensor, incx, resultTensor);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasIsamin error.";
//         return -1;
//     }
//     return 0;
// }

uint8_t _asdBlasIcamax(uint64_t &handle, const int64_t n, torch::Tensor &A, const int64_t incx, torch::Tensor &result)
{
    aclTensor *ATensor = torchTensor2aclTensor(A);
    aclTensor *resultTensor = torchTensor2aclTensor(result);

    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasIcamax(_handle, n, ATensor, incx, resultTensor);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasIcamax error.";
        return -1;
    }
    return 0;
}

// uint8_t _asdBlasIcamin(uint64_t &handle, const int64_t n, torch::Tensor &A, const int64_t incx, torch::Tensor &result)
// {
//     Mki::Tensor ATensor = torchTensor2AsdTensor(A);
//     Mki::Tensor resultTensor = torchTensor2AsdTensor(result);

//     asdBlasHandle _handle = (void *)handle;

//     AspbStatus status = asdBlasIcamin(_handle, n, ATensor, incx, resultTensor);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasIcamin error.";
//         return -1;
//     }
//     return 0;
// }

uint8_t _asdBlasMakeCaxpyPlan(uint64_t handle)
{
    asdBlasHandle _handle = (void *)handle;
    AspbStatus status = asdBlasMakeCaxpyPlan(_handle);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasMakeCaxpyPlan error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCaxpy(uint64_t &handle, const int64_t n, const std::complex<float> alpha, torch::Tensor &x,
                      const int64_t incx, torch::Tensor &y, const int64_t incy)
{
    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCaxpy(_handle, n, alpha, xAsdTensor, incx, yAsdTensor, incy);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCaxpy error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasHCgemvBatched(uint64_t &handle, const char trans, const int64_t m, const int64_t n,
                    const std::complex<op::fp16_t> alpha, torch::Tensor &A, const int64_t lda,
                    torch::Tensor &x, const int64_t incx, const std::complex<op::fp16_t> beta,
                    torch::Tensor &y, const int64_t incy, const int64_t batchCount)
{
    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
    }

    aclTensor *xAsdTensor = torchF16Tensor2aclC32Tensor(x);
    aclTensor *yAsdTensor = torchF16Tensor2aclC32Tensor(y);
    aclTensor *aAsdTensor = torchF16Tensor2aclC32Tensor(A);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasHCgemvBatched(_handle, transValue->second, m, n, alpha,
                    aAsdTensor, lda, xAsdTensor, incx, beta, yAsdTensor, incy, batchCount);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasHCgemvBatched error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasCgemvBatched(uint64_t &handle, const char trans, const int64_t m, const int64_t n,
                    const std::complex<float> alpha, torch::Tensor &A, const int64_t lda,
                    torch::Tensor &x, const int64_t incx, const std::complex<float> beta,
                    torch::Tensor &y, const int64_t incy, const int64_t batchCount)
{
    auto transValue = OPERATION_MAP.find(trans);
    if (transValue == OPERATION_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
    }

    aclTensor *xAsdTensor = torchTensor2aclTensor(x);
    aclTensor *yAsdTensor = torchTensor2aclTensor(y);
    aclTensor *aAsdTensor = torchTensor2aclTensor(A);
    asdBlasHandle _handle = (void *)handle;

    AspbStatus status = asdBlasCgemvBatched(_handle, transValue->second, m, n, alpha,
                    aAsdTensor, lda, xAsdTensor, incx, beta, yAsdTensor, incy, batchCount);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasCgemvBatched error.";
        return -1;
    }
    return 0;
}

// uint8_t _asdBlasMakeCsyr2kPlan(uint64_t handle, const char uplo, const int64_t n, const int64_t k)
// {
//     asdBlasHandle _handle = (void *)handle;

//     auto uploValue = FILL_MODE_MAP.find(uplo);
//     if (uploValue == FILL_MODE_MAP.end()) {
//         ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
//     }

//     AspbStatus status = asdBlasMakeCsyr2kPlan(_handle, uploValue->second, n, k);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasMakeCsyr2kPlan error.";
//         return -1;
//     }
//     return 0;
// }

// uint8_t _asdBlasCsyr2k(uint64_t &handle, const char uplo, const char trans, const int64_t n, const int64_t k,
//                        const std::complex<float> alpha, torch::Tensor &A, const int64_t lda, torch::Tensor &B,
//                        const int64_t ldb, const std::complex<float> beta, torch::Tensor &C, const int64_t ldc)
// {
//     asdBlasHandle _handle = (void *)handle;
    
//     auto uploValue = FILL_MODE_MAP.find(uplo);
//     if (uploValue == FILL_MODE_MAP.end()) {
//         ASDSIP_LOG(ERROR) << "not support uplo value:" << uplo;
//     }
//     auto transValue = OPERATION_MAP.find(trans);
//     if (transValue == OPERATION_MAP.end()) {
//         ASDSIP_LOG(ERROR) << "not support trans value:" << trans;
//     }

//     Mki::Tensor asdTensorA = torchTensor2AsdTensor(A);
//     Mki::Tensor asdTensorB = torchTensor2AsdTensor(B);
//     Mki::Tensor asdTensorC = torchTensor2AsdTensor(C);

//     AspbStatus status = asdBlasCsyr2k(_handle, uploValue->second, transValue->second, n, k, &alpha, asdTensorA,
//                                      lda, asdTensorB, ldb, beta, asdTensorC, ldc);

//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasCsyr2k error.";
//         return -1;
//     }
//     return 0;
// }

// uint8_t _asdBlasMakeCaxpyV2Plan(uint64_t handle, int64_t n, const std::complex<float> alpha, uint64_t workspace) {

//     asdBlasHandle _handle = (void *)handle;
//     uint8_t *workspaceBuffer = (uint8_t *)workspace;
//     AspbStatus status = asdBlasMakeCaxpyV2Plan(_handle, n, alpha, workspaceBuffer);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasMakeCaxpyV2Plan error.";
//         return -1;
//     }
//     return 0;
// }

// uint8_t _asdBlasCaxpyV2(uint64_t &handle, const int64_t n,
//                          const std::complex<float> alpha,
//                          torch::Tensor &x, const int64_t incx,
//                          torch::Tensor &y, const int64_t incy) {
//     Mki::Tensor xAsdTensor = torchTensor2AsdTensor(x);
//     Mki::Tensor yAsdTensor = torchTensor2AsdTensor(y);
//     asdBlasHandle _handle = (void *)handle;

//     AspbStatus status = asdBlasCaxpyV2(_handle, n, alpha, xAsdTensor, incx, yAsdTensor, incy);
//     if (status != AsdSip::ErrorType::ACL_SUCCESS) {
//         ASDSIP_LOG(ERROR) << "_asdBlasCaxpyV2 error.";
//         return -1;
//     }
//     return 0;
// }

int64_t _asdBlasGetWorkspaceSize(uint64_t handle)
{
    asdBlasHandle handle_ = (void *)handle;
    size_t work_size;
    AspbStatus status = asdBlasGetWorkspaceSize(handle_, work_size);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "asdBlasGetWorkspaceSize error.";
        return -1;
    }
    return work_size;
}

uint8_t _asdBlasSetWorkspace(uint64_t handle, uint64_t workspace)
{
    asdBlasHandle handle_ = (void *)handle;
    uint8_t *workspaceBuffer = (uint8_t *)workspace;
    AspbStatus status = asdBlasSetWorkspace(handle_, workspaceBuffer);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasDestroy error.";
        return -1;
    }
    return 0;
}

uint8_t _asdBlasDestroy(uint64_t handle)
{
    asdBlasHandle handle_ = (void *)handle;
    AspbStatus status = asdBlasDestroy(handle_);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "_asdBlasDestroy error.";
        return -1;
    }
    return 0;
}

uint8_t _interpolation(const torch::Tensor &inTensor, const torch::Tensor &tabTensor, const torch::Tensor &posFloor,
                       const torch::Tensor &posTabIndex, const int64_t interpNum, const int64_t quantNum,
                       const int64_t interpLength, torch::Tensor &outTensor)
{
    aclTensor* inAsdTensor = torchTensor2aclTensor(inTensor);
    aclTensor* tabAsdTensor = torchTensor2aclTensor(tabTensor);
    aclTensor* posFloorAsdTensor = torchTensor2aclTensor(posFloor);
    aclTensor* posTabIndexAsdTensor = torchTensor2aclTensor(posTabIndex);
    aclTensor* outAsdTensor = torchTensor2aclTensor(outTensor);

    MkiRtStream stream = getCurrentStream();

    AspbStatus status = rsInterpolationBySinc(inAsdTensor, tabAsdTensor, posFloorAsdTensor, posTabIndexAsdTensor,
                                              outAsdTensor, interpNum, quantNum, interpLength, stream);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Something error.";
        return -1;
    }

    return 0;
}

uint8_t _interpolationCpu(const torch::Tensor &inTensor, const torch::Tensor &posTensor, const int64_t batch,
                          const int64_t signalLength, const int64_t interpLength, torch::Tensor &outTensor)
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


size_t _asdConvolveGetWorkspaceSize(int64_t signalLen, int64_t kernelLen)
{
    size_t work_size;
    AspbStatus status = asdConvolveGetWorkspaceSize(signalLen, kernelLen, work_size);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "asdConvolveGetWorkspaceSize error.";
        return -1;
    }
    return work_size;

}

uint8_t _asdConvolve(const torch::Tensor &signal, const torch::Tensor &kernel, torch::Tensor &output,
    std::string mode, uint64_t workspace)
{

    auto modeVal = CONVOLVE_MODE_MAP.find(mode);
    if (modeVal == CONVOLVE_MODE_MAP.end()) {
        ASDSIP_LOG(ERROR) << "not support mode value:" << mode;
    }
    
    aclTensor* signalAsdTensor;
    aclTensor* kernelAsdTensor;
    aclTensor* outputAsdTensor;

    if (kernel.scalar_type() == torch::ScalarType::Half) {
        signalAsdTensor = torchF16Tensor2aclC32Tensor(signal);
        kernelAsdTensor = torchTensor2aclTensor(kernel);
        outputAsdTensor = torchF16Tensor2aclC32Tensor(output);
    } else {
        signalAsdTensor = torchTensor2aclTensor(signal);
        kernelAsdTensor = torchTensor2aclTensor(kernel);
        outputAsdTensor = torchTensor2aclTensor(output);

    }

    MkiRtStream stream = getCurrentStream();

    AspbStatus status = asdConvolve(signalAsdTensor, kernelAsdTensor, outputAsdTensor, modeVal->second,
        stream, (uint8_t *)(workspace));

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Something error.";
        return -1;
    }

    return 0;
}

uint8_t _interpbycoeff(const torch::Tensor &inTensor, const torch::Tensor &coeffTensor, torch::Tensor &outTensor)
{
    aclTensor* inAsdTensor = torchTensor2aclTensor(inTensor);
    aclTensor* coeffAsdTensor = torchTensor2aclTensor(coeffTensor);
    aclTensor* outAsdTensor = torchTensor2aclTensor(outTensor);

    MkiRtStream stream = getCurrentStream();
    void *buffer = nullptr;
    MkiRtMemMallocDevice((void **)&buffer, 65536 * 20 * 2, MKIRT_MEM_DEFAULT);
    AspbStatus status = asdInterpWithCoeff(inAsdTensor, coeffAsdTensor, outAsdTensor, stream, buffer);

    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        ASDSIP_LOG(ERROR) << "Something error.";
        return -1;
    }

    return 0;
}


PYBIND11_MODULE(libasdsip, m)
{
    m.def("setDevId", &_setDevId, "function of setting device ID for global use.");
    m.def("setDevice", &_setDevice, "function of setting device before current operator execution.");
    m.def("resetDevice", &_resetDevice, "function of resetting device after current operator execution.");
    // m.def("transpose", &_transpose, "function dispatching to transpose operator.");
    m.def("swapLast2Axes", &_swapLast2Axes, "function dispatching to swapLast2Axes operator.");
    m.def("swapLast2AxesGetWorkspaceSize", &_swapLast2AxesGetWorkspaceSize,
          "function of estimating workspace for swapLast2Axes operator.");
    m.def("asdMul", &_asdMul, "function dispatching to asdMul operator.");
    // m.def("conj", &_conj, "function dispatching to conj operator.");
    // m.def("modulus_v2", &_modulus, "function dispatching to modulus operator.");
    // m.def("sub", &_sub, "function dispatching to sub operator.");
    // m.def("add", &_add, "function dispatching to add operator.");
    // m.def("matmul", &_matMul, "function dispatching to matmul operator.");
    // m.def("muls", &_muls, "function dispatching to muls operator.");
    // m.def("cast", &_cast, "function dispatching to cast operator.");
    // m.def("mul", &_mul, "function dispatching to mul operator.");
    // m.def("zeros_like", &_zerolike, "function dispatching to zeros_like operator.");
    // m.def("as_strided", &_asStrided, "function dispatching to as_strided operator.");
    // m.def("complex_separation", &_complex_separation, "function dispatching to complex_separation operator.");
    // m.def("complex_combination", &_complex_combination, "function dispatching to complex_combination operator.");
    // m.def("ssyr2", &_ssyr2, "function dispatching to ssyr2 operator.");
    // m.def("ssyr2v2", &_ssyr2v2, "function dispatching to ssyr2v2 operator.");
    // m.def("ssyr2_v3", &_ssyr2_v3, "function dispatching to ssyr2v3 operator.");
    // m.def("strmv", &_strmv, "function dispatching to strmv operator.");
    // m.def("strmv_v2", &_strmv_v2, "function dispatching to strmv_v2 operator.");
    // m.def("strmm", &_strmm, "function dispatching to strmm operator.");
    // m.def("strmmV2", &_strmmV2, "function dispatching to strmmV2 operator.");
    // m.def("strmm_v3", &_strmm_v3, "function dispatching to strmm_v3 operator.");
    // m.def("strsm", &_strsm, "function dispatching to strsm operator.");
    // m.def("ctrmv", &_ctrmv, "function dispatching to ctrmv operator.");
    // m.def("ctrmv_v2", &_ctrmv_v2, "function dispatching to ctrmv_v2 operator.");
    // m.def("cgemm", &_cgemm, "function dispatching to cgemm operator.");
    // m.def("sgetrf", &_sgetrf, "function dispatching to sgetrf operator.");
    // m.def("sgetri", &_sgetri, "function dispatching to sgetri operator.");
    // m.def("inverse", &_inverse, "function dispatching to inverse operator.");
    m.def("fftInit", &_fftInit, "function of initializing the operator fft.");
    m.def("fftInit1DCol", &_fftInit1DCol, "function of initializing the operator fft1dCol.");
    m.def("fftExecute", &_fftExecute, "function dispatching to fft operator.");
    m.def("fftFinalize", &_fftFinalize, "function of finalizing the operator fft.");
    m.def("fftInit2D", &_fftInit2D, "function of initializing the operator fft2d.");
    m.def("fftGetWorkspaceSize", &_fftGetWorkspaceSize, "function of estimating workspace for fft operator.");
    m.def("fftSetWorkspace", &_fftSetWorkspace, "function of setting workspace for the fft operator.");
    // m.def("PostCFARDetection", &_PostCFARDetection, "function dispatching to PostCFARDetection operator.");
    // m.def("padZero", &_padZero, "function dispatching to padZero operator.");
    // m.def("cutZero", &_cutZero, "function dispatching to cutZero operator.");
    m.def("asdBlasCreateHandle", &_asdBlasCreateHandle, "function dispatching to create blas handle.");
    m.def("asdBlasMakeSsyr2Plan", &_asdBlasMakeSsyr2Plan, "function dispatching to create op named ssyr2 plan.");
    m.def("asdBlasMakeCgemmPlan", &_asdBlasMakeCgemmPlan, "function dispatching to create op named cgemm plan.");
    m.def("asdMakeBlasStrmvPlan", &_asdMakeBlasStrmvPlan, "function dispatching to create op named strmv plan.");
    m.def("asdBlasMakeCgemvPlan", &_asdBlasMakeCgemvPlan, "function dispatching to create op named cgemv plan.");
    m.def("asdBlasMakeAsumPlan", &_asdBlasMakeAsumPlan, "function dispatching to create op named asum plan.");
    m.def("asdBlasMakeNrm2Plan", &_asdBlasMakeNrm2Plan, "function dispatching to create op named nrm2 plan.");
    m.def("asdBlasMakeCopyPlan", &_asdBlasMakeCopyPlan, "function dispatching to create op named copy plan.");
    m.def("asdBlasMakeCalPlan", &_asdBlasMakeCalPlan, "function dispatching to create op named cal plan.");
    m.def("asdBlasMakeCtrmvPlan", &_asdBlasMakeCtrmvPlan, "function dispatching to create op named ctrmv plan.");
    m.def("asdBlasMakeCgercPlan", &_asdBlasMakeCgercPlan, "function dispatching to create op named cgerc plan.");
    m.def("asdBlasMakeDotPlan", &_asdBlasMakeDotPlan, "function dispatching to create op named dot plan.");
    m.def("asdBlasMakeRotPlan", &_asdBlasMakeRotPlan, "function dispatching to create op named rot plan.");
    m.def("asdBlasMakeSsyrPlan", &_asdBlasMakeSsyrPlan, "function dispatching to create op named ssyr plan.");
    m.def("asdBlasMakeStrmmPlan", &_asdBlasMakeStrmmPlan, "function dispatching to create op named strmm plan.");
    // m.def("asdBlasMakeCsyrkPlan", &_asdBlasMakeCsyrkPlan, "function dispatching to create op named csyrk plan.");
    m.def("asdBlasMakeColwiseMulPlan", &_asdBlasMakeColwiseMulPlan,
          "function dispatching to create op named colwise_mul plan.");
    m.def("asdBlasMakeComplexMatDotPlan", &_asdBlasMakeComplexMatDotPlan,
          "function dispatching to create op named complex_mat_dot plan.");
    // m.def("asdBlasMakeSmatinvBatchedPlan", &_asdBlasMakeSmatinvBatchedPlan, "function dispatching to create op named smatinv plan.");
    // m.def("asdBlasMakeCmatinvBatchedPlan", &_asdBlasMakeCmatinvBatchedPlan, "function dispatching to create op named CmatinvBatched plan.");
    m.def("asdBlasMakeIamaxPlan", &_asdBlasMakeIamaxPlan, "function dispatching to create op named iamax plan.");
    // m.def("asdBlasMakeIaminPlan", &_asdBlasMakeIaminPlan, "function dispatching to create op named iamin plan.");
    m.def("asdBlasMakeCaxpyPlan", &_asdBlasMakeCaxpyPlan, "function dispatching to create op named caxpy plan.");
    // m.def("asdBlasMakeCaxpyV2Plan", &_asdBlasMakeCaxpyV2Plan, "function dispatching to create op named caxpy_v2 plan.");
    m.def("asdBlasMakeSwapPlan", &_asdBlasMakeSwapPlan, "function dispatching to create op named swap plan.");
    // m.def("asdBlasMakeCsyr2kPlan", &_asdBlasMakeCsyr2kPlan, "function dispatching to create op named csyr2k plan.");
    m.def("asdBlasMakeHCgemvBatchedPlan", &_asdBlasMakeHCgemvBatchedPlan,
            "function dispatching to create op named hcgemv batched plan.");
    m.def("asdBlasMakeCgemvBatchedPlan", &_asdBlasMakeCgemvBatchedPlan,
            "function dispatching to create op named cgemv batched plan.");
    m.def("asdBlasDestroy", &_asdBlasDestroy, "function dispatching to create blas handle.");
    m.def("asdBlasSetStream", &_asdBlasSetStream, "function dispatching to bind both stream and handle.");
    m.def("asdBlasSsyr", &_asdBlasSsyr, "function dispatching to ssyr operator.");
    m.def("asdBlasSsyr2", &_asdBlasSsyr2, "function dispatching to ssyr2 operator.");
    m.def("asdBlasCgemm", &_asdBlasCgemm, "function dispatching to cgemm operator.");
    m.def("asdBlasStrmm", &_asdBlasStrmm, "function dispatching to strmm operator.");
    m.def("asdBlasStrmv", &_asdBlasStrmv, "function dispatching to strmv operator.");
    m.def("asdBlasCgemv", &_asdBlasCgemv, "function dispatching to cgemv operator.");
    m.def("asdBlasSasum", &_asdBlasSasum, "function dispatching to sasum operator.");
    m.def("asdBlasScasum", &_asdBlasScasum, "function dispatching to scasum operator.");
    m.def("asdBlasSnrm2", &_asdBlasSnrm2, "function dispatching to snrm2 operator.");
    m.def("asdBlasScnrm2", &_asdBlasScnrm2, "function dispatching to scnrm2 operator.");
    m.def("asdBlasScopy", &_asdBlasScopy, "function dispatching to scopy operator.");
    m.def("asdBlasCcopy", &_asdBlasCcopy, "function dispatching to ccopy operator.");
    m.def("asdBlasSscal", &_asdBlasSscal, "function dispatching to sscal operator.");
    m.def("asdBlasCscal", &_asdBlasCscal, "function dispatching to cscal operator.");
    m.def("asdBlasCsscal", &_asdBlasCsscal, "function dispatching to csscal operator.");
    m.def("asdBlasCtrmv", &_asdBlasCtrmv, "function dispatching to ctrmv operator.");
    m.def("asdBlasCgerc", &_asdBlasCgerc, "function dispatching to cgerc operator.");
    m.def("asdBlasSdot", &_asdBlasSdot, "function dispatching to sdot operator.");
    m.def("asdBlasCdotu", &_asdBlasCdotu, "function dispatching to cdotu operator.");
    m.def("asdBlasCdotc", &_asdBlasCdotc, "function dispatching to cdotc operator.");
    m.def("asdBlasColwiseMul", &_asdBlasColwiseMul, "function dispatching to colwise_mul operator.");
    m.def("asdBlasComplexMatDot", &_asdBlasComplexMatDot, "function dispatching to cpmlex_mat_dot operator.");
    m.def("asdBlasCsrot", &_asdBlasCsrot, "function dispatching to csrot operator.");
    // m.def("asdBlasSmatinvBatched", &_asdBlasSmatinvBatched, "function dispatching to smatinv_batched operator.");
    // m.def("asdBlasCmatinvBatched", &_asdBlasCmatinvBatched, "function dispatching to cmatinv_batched operator.");
    m.def("asdBlasIsamax", &_asdBlasIsamax, "function dispatching to iamax operator.");
    m.def("asdBlasIcamax", &_asdBlasIcamax, "function dispatching to iamax operator.");
    // m.def("asdBlasIsamin", &_asdBlasIsamin, "function dispatching to iamin operator.");
    // m.def("asdBlasIcamin", &_asdBlasIcamin, "function dispatching to iamin operator.");
    m.def("asdBlasCaxpy", &_asdBlasCaxpy, "function dispatching to caxpy operator.");
    // m.def("asdBlasCsyrk", &_asdBlasCsyrk, "function dispatching to csyrk operator.");
    // // m.def("asdBlasCaxpyV2", &_asdBlasCaxpyV2, "function dispatching to caxpy_v2 operator.");
    m.def("setBlasWorkspace", &_setBlasWorkspace, "function dispatching to malloc blas workspace.");
    m.def("interpolation", &_interpolation, "function dispatching to interpolation operator.");
    m.def("interpolationCpu", &_interpolationCpu, "function dispatching to interpolation operator.");
    m.def("asdBlasSswap", &_asdBlasSswap, "function dispatching to sswap operator.");
    m.def("asdBlasCswap", &_asdBlasCswap, "function dispatching to cswap operator.");
    // m.def("asdBlasCsyr2k", &_asdBlasCsyr2k, "function dispatching to csyr2k operator.");
    m.def("asdBlasHCgemvBatched", &_asdBlasHCgemvBatched, "function dispatching to hcgemv batched operator.");
    m.def("asdBlasCgemvBatched", &_asdBlasCgemvBatched, "function dispatching to cgemv batched operator.");
    m.def("asdBlasGetWorkspaceSize", &_asdBlasGetWorkspaceSize, "function of get workspace size for blas operator.");
    m.def("asdBlasSetWorkspace", &_asdBlasSetWorkspace, "function of set workspace size for blas operator.");
    m.def("asdConvolveGetWorkspaceSize", &_asdConvolveGetWorkspaceSize, "function of asdConvolveGetWorkspaceSize.");
    m.def("asdConvolve", &_asdConvolve, "function of asdConvolve.");
    m.def("interpbycoeff", &_interpbycoeff, "function dispatching to interpolation operator.");
    m.def("asdBlasMakeHCgemmBatchedPlan", &_asdBlasMakeHCgemmBatchedPlan, "make plan for hcgemm.");
    m.def("asdBlasHCgemmBatched", &_asdBlasHCgemmBatched, "run hcgemm.");
    m.def("asdBlasMakeCgemmBatchedPlan", &_asdBlasMakeCgemmBatchedPlan, "make plan for cgemm.");
    m.def("asdBlasCgemmBatched", &_asdBlasCgemmBatched, "run cgemm.");
}
