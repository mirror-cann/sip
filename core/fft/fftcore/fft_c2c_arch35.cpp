/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <mki/utils/rt/rt.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ops.h"
#include "utils/ops_base.h"
#include "fft_c2c_arch35.h"
#include "fftcore/fft_c2c_arch35_core.h"

constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = 2 * K_PI;

using namespace AsdSip;

size_t FftC2CCoreArch35::EstimateWorkspaceSize()
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        int64_t singleBufferSize = static_cast<int64_t>(problemDesc.batch) * static_cast<int64_t>(problemDesc.nDoing) * 2 * sizeof(float);
        int64_t workspaceSize = 2 * singleBufferSize;
        ASDSIP_LOG(INFO) << "ASCEND_950 FftC2CCoreArch35 workspace size: " << workspaceSize;
        return static_cast<size_t>(workspaceSize);
    }
    const KernelInfo &kernelInfo = kernel->GetKernelInfo();
    return getAlignedSize(kernelInfo.GetTotalScratchSize());
}

void FftC2CCoreArch35::Run(void *input, void *output, void *stream, workspace::Workspace &workspace)
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        FftOperation::Run(input, output, stream, workspace);
        ASDSIP_LOG(INFO) << "ASCEND_950 FftC2CCoreArch35 run success.";
    }
}

void FftC2CCoreArch35::DestroyInDevice() const
{
    uint8_t *deviceLaunchBuffer = runInfo.GetTilingDeviceAddr();
    if (deviceLaunchBuffer != nullptr) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
    }
}

void FftC2CCoreArch35::InitRadix()
{
    plan.clear();
    int64_t tempN = static_cast<int64_t>(problemDesc.nDoing);

    if (tempN <= 0) {
        ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 nDoing must be positive, got " << tempN;
        throw std::runtime_error("FftC2CCoreArch35 nDoing must be positive.");
    }

    constexpr size_t numRadices = std::size(ALLOWED_RADICES);

    while (tempN > 1) {
        int64_t radix = 0;
        for (size_t i = 0; i < numRadices; i++) {
            if (tempN % ALLOWED_RADICES[i] == 0) {
                radix = ALLOWED_RADICES[i];
                break;
            }
        }

        if (radix == 0) {
            std::string allowedStr;
            for (size_t i = 0; i < numRadices; i++) {
                if (i > 0) allowedStr += ", ";
                allowedStr += std::to_string(ALLOWED_RADICES[i]);
            }
            ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 nDoing contains prime factors other than "
                              << allowedStr << ". Remaining: " << tempN;
            throw std::runtime_error("FftC2CCoreArch35: unsupported signal length. "
                                     "Only prime factors " + allowedStr + " are allowed.");
        }

        int64_t M = tempN / radix;
        plan.push_back({radix, M});
        tempN = M;
    }

    ASDSIP_LOG(INFO) << "FftC2CCoreArch35 init radix success. plan size: " << plan.size();
}

AspbStatus FftC2CCoreArch35::BuildFftPlan()
{
    if (plan.empty()) {
        ASDSIP_LOG(ERROR) << "FftC2CCoreArch35 plan is empty, InitRadix not called?";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }

    double sign = problemDesc.forward ? (-1.0) : (1.0);
    size_t planLen = plan.size();
    size_t radixListSize = planLen * sizeof(int32_t);

    auto radixListFunc = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor *t = new AsdSip::FFTensor;
        int32_t *host = new int32_t[planLen];
        for (size_t s = 0; s < planLen; s++) {
            host[s] = static_cast<int32_t>(plan[s].radix);
        }
        t->desc = {Mki::TensorDType::TENSOR_DTYPE_INT32, Mki::TensorFormat::TENSOR_FORMAT_ND,
                   {static_cast<int64_t>(planLen)}, {}, 0};
        t->hostData = host;
        t->dataSize = radixListSize;
        return t;
    };
    AsdSip::CoeffKey radixKey = {coreType, 100, {static_cast<int64_t>(problemDesc.nDoing)}, problemDesc.forward};
    radixListTensor = FFTensorCache::getCoeff(radixKey, radixListFunc);

    size_t totalDftFloats = 0;
    size_t totalTwFloats = 0;
    for (size_t s = 0; s < planLen; s++) {
        int64_t radix = plan[s].radix;
        int64_t M = plan[s].M;
        totalDftFloats += 2 * radix * radix;
        totalTwFloats += 2 * radix * M;
    }

    size_t dftDataSize = totalDftFloats * sizeof(float);

    auto dftFunc = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor *t = new AsdSip::FFTensor;
        float *host = nullptr;
        try {
            host = new float[totalDftFloats]();
        } catch (std::bad_alloc& e) {
            delete t;
            ASDSIP_LOG(ERROR) << "dftMatrixArray host malloc failed";
            throw std::runtime_error("dftMatrixArray host malloc failed.");
        }

        size_t offset = 0;
        for (size_t s = 0; s < planLen; s++) {
            int64_t radix = plan[s].radix;
            for (int64_t u = 0; u < radix; u++) {
                for (int64_t v = 0; v < radix; v++) {
                    double angle = sign * K_2PI * u * v / radix;
                    host[offset + (2 * u) * radix + v] = static_cast<float>(cos(angle));
                    host[offset + (2 * u + 1) * radix + v] = static_cast<float>(sin(angle));
                }
            }
            offset += 2 * radix * radix;
        }

        t->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND,
                   {static_cast<int64_t>(totalDftFloats)}, {}, 0};
        t->hostData = host;
        t->dataSize = dftDataSize;
        return t;
    };

    size_t twDataSize = totalTwFloats * sizeof(float);

    auto twFunc = [=]() -> AsdSip::FFTensor* {
        AsdSip::FFTensor *t = new AsdSip::FFTensor;
        float *host = nullptr;
        try {
            host = new float[totalTwFloats]();
        } catch (std::bad_alloc& e) {
            delete t;
            ASDSIP_LOG(ERROR) << "twMatrixArray host malloc failed";
            throw std::runtime_error("twMatrixArray host malloc failed.");
        }

        size_t offset = 0;
        int64_t tempN = static_cast<int64_t>(problemDesc.nDoing);

        for (size_t s = 0; s < planLen; s++) {
            int64_t radix = plan[s].radix;
            int64_t M = plan[s].M;

            for (int64_t k1 = 0; k1 < radix; k1++) {
                for (int64_t n2 = 0; n2 < M; n2++) {
                    double angle = sign * K_2PI * k1 * n2 / tempN;
                    host[offset + (2 * k1) * M + n2] = static_cast<float>(cos(angle));
                    host[offset + (2 * k1 + 1) * M + n2] = static_cast<float>(sin(angle));
                }
            }
            offset += 2 * radix * M;
            tempN = M;
        }

        t->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND,
                   {static_cast<int64_t>(totalTwFloats)}, {}, 0};
        t->hostData = host;
        t->dataSize = twDataSize;
        return t;
    };

    AsdSip::CoeffKey dftKey = {coreType, 101, {static_cast<int64_t>(problemDesc.nDoing)}, problemDesc.forward};
    dftMatrixArray = FFTensorCache::getCoeff(dftKey, dftFunc);

    AsdSip::CoeffKey twKey = {coreType, 102, {static_cast<int64_t>(problemDesc.nDoing)}, problemDesc.forward};
    twMatrixArray = FFTensorCache::getCoeff(twKey, twFunc);

    ASDSIP_LOG(INFO) << "FftC2CCoreArch35 BuildFftPlan success. "
                     << "planLen=" << planLen
                     << ", dftFloats=" << totalDftFloats
                     << ", twFloats=" << totalTwFloats;

    return AsdSip::ErrorType::ACL_SUCCESS;
}

bool FftC2CCoreArch35::PreAllocateInDevice()
{
    if (BuildFftPlan() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    return true;
}

AspbStatus FftC2CCoreArch35::InitTactic()
{
    if (Mki::PlatformInfo::Instance().GetPlatformType() == Mki::PlatformType::ASCEND_950) {
        ASDSIP_LOG(INFO) << "ASCEND_950 FftC2CCoreArch35 init tactic";
    }

    OpParam::FftC2CArch35 param = {problemDesc.nDoing,
                                    problemDesc.batch,
                                    static_cast<int64_t>(plan.size()),
                                    1 - int(problemDesc.forward)};
    ASDSIP_LOG(DEBUG) << "OpDesc info: " << param.ToString();

    Tensor tensorIn;
    Tensor tensorOut;
    int64_t inputN = problemDesc.nDoing;
    tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, inputN}, {}, 0};
    tensorIn.dataSize = static_cast<size_t>(problemDesc.batch) * static_cast<size_t>(inputN)
                        * GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);

    tensorOut.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {problemDesc.batch, inputN}, {}, 0};
    tensorOut.dataSize = static_cast<size_t>(problemDesc.batch) * static_cast<size_t>(inputN)
                         * GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);

    launchParam.SetParam(param);
    launchParam.AddInTensor(tensorIn);
    launchParam.AddInTensor(*dftMatrixArray);
    launchParam.AddInTensor(*twMatrixArray);
    launchParam.AddInTensor(*radixListTensor);
    launchParam.AddOutTensor(tensorOut);

    Operation *op = Ops::Instance().GetOperationByName(opName);
    if (op == nullptr) {
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    op->InferShape(launchParam);
    kernel = std::unique_ptr<Kernel>(op->GetBestKernel(launchParam));
    ASDSIP_ECHECK(kernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    uint8_t *deviceLaunchBuffer = nullptr;
    kernel->SetLaunchWithTiling(false);
    uint32_t launchBufferSize = kernel->GetTilingSize(launchParam);
    ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    uint8_t hostLaunchBuffer[launchBufferSize];
    kernel->SetTilingHostAddr(hostLaunchBuffer, launchBufferSize);
    kernel->Init(launchParam);

    void* tempDevicePtr = nullptr;
    int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
    ASDSIP_ECHECK(st == MKIRT_SUCCESS, "malloc device memory fail", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

    deviceLaunchBuffer = static_cast<uint8_t *>(tempDevicePtr);
    st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer, launchBufferSize,
                      MKIRT_MEMCOPY_HOST_TO_DEVICE);
    if (st != MKIRT_SUCCESS) {
        MkiRtMemFreeDevice(deviceLaunchBuffer);
        deviceLaunchBuffer = nullptr;
        ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "copy host memory to device fail";
        return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
    }
    runInfo.SetTilingDeviceAddr(deviceLaunchBuffer);

    ASDSIP_LOG(INFO) << "FftC2CCoreArch35 init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
