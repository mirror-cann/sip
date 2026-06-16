/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <cmath>
#include <vector>
#include <mki/utils/rt/rt.h>
#include <mki/utils/platform/platform_info.h>
#include "utils/assert.h"
#include "log/log.h"
#include "ops.h"
#include "utils/ops_base.h"
#include "params/fft_c2c_arch35.h"
#include "../../../ops/fft/common/include/tiling/fft_all_mix_tiling_data.h"
#include "fftcore/fft_c2c2d_arch35_core.h"
#include "params/fft_c2c2d_arch35.h"

constexpr double K_PI = 3.14159265358979323846;
constexpr double K_2PI = K_PI * 2;

using namespace AsdSip;

namespace {
constexpr int64_t MIXED_RADIX_THRESHOLD = 1 << 14;
constexpr int64_t MULTI_CORE_BATCH_THRESHOLD = 32;

int32_t Log2PowerOfTwo(int64_t value)
{
    int32_t log = 0;
    while (value > 1) {
        value >>= 1;
        ++log;
    }
    return log;
}

bool IsPowerOfTwo(int64_t value)
{
    return value > 1 && (value & (value - 1)) == 0;
}

bool ShouldUseMixedRadixKernel(int64_t fftLen, int64_t batchSize)
{
    return !IsPowerOfTwo(fftLen) || fftLen < MIXED_RADIX_THRESHOLD || batchSize <= MULTI_CORE_BATCH_THRESHOLD;
}

void FreeDeviceAddrs(std::vector<uint8_t *> &deviceAddrs)
{
    for (uint8_t *deviceAddr : deviceAddrs) {
        if (deviceAddr != nullptr) {
            MkiRtMemFreeDevice(deviceAddr);
        }
    }
    deviceAddrs.clear();
}

size_t GetRadix2TwiddleFloatCount(int64_t n)
{
    size_t totalTwComplex = 0;
    for (int64_t len = 2; len <= n; len <<= 1) {
        totalTwComplex += static_cast<size_t>(len >> 1);
    }
    return totalTwComplex * 2;
}

AsdSip::FFTensor *BuildRadix2TwiddleTensor(int64_t n, bool forward)
{
    double sign = forward ? -1.0 : 1.0;
    size_t totalTwFloats = GetRadix2TwiddleFloatCount(n);
    float *host = new float[totalTwFloats]();

    size_t offset = 0;
    for (int64_t len = 2; len <= n; len <<= 1) {
        int64_t half = len >> 1;
        for (int64_t j = 0; j < half; j++) {
            double angle = sign * K_2PI * j / len;
            host[(offset + j) * 2] = static_cast<float>(std::cos(angle));
            host[(offset + j) * 2 + 1] = static_cast<float>(std::sin(angle));
        }
        offset += static_cast<size_t>(half);
    }

    AsdSip::FFTensor *tensor = new AsdSip::FFTensor;
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND,
        {static_cast<int64_t>(totalTwFloats)}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = totalTwFloats * sizeof(float);
    return tensor;
}

std::vector<int32_t> FactorizeMixed(int64_t n)
{
    std::vector<int32_t> radices;
    constexpr int32_t allowedRadices[] = {2, 3, 5, 7, 11, 13, 17, 19};
    for (int32_t radix : allowedRadices) {
        while (n % radix == 0) {
            radices.push_back(radix);
            n /= radix;
        }
    }
    if (n != 1) {
        ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 unsupported mixed-radix factor remains: " << n;
        throw std::runtime_error("FftC2C2DCoreArch35 unsupported mixed-radix factorization.");
    }
    return radices;
}

AsdSip::FFTensor *BuildRadixListTensor(const std::vector<int32_t> &radices)
{
    AsdSip::FFTensor *tensor = new AsdSip::FFTensor;
    int32_t *host = new int32_t[radices.size()];
    for (size_t i = 0; i < radices.size(); ++i) {
        host[i] = radices[i];
    }
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_INT32, Mki::TensorFormat::TENSOR_FORMAT_ND,
        {static_cast<int64_t>(radices.size())}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = radices.size() * sizeof(int32_t);
    return tensor;
}

AsdSip::FFTensor *BuildRadix2DftTensor()
{
    AsdSip::FFTensor *tensor = new AsdSip::FFTensor;
    float *host = new float[1]();
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND, {1}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = sizeof(float);
    return tensor;
}

AsdSip::FFTensor *BuildMixedTwiddleTensor(const std::vector<int32_t> &radices, bool forward)
{
    double sign = forward ? -1.0 : 1.0;
    int64_t totalTwComplex = 0;
    int64_t prev = 1;
    for (int32_t radix : radices) {
        totalTwComplex += radix * prev;
        prev *= radix;
    }

    float *host = new float[static_cast<size_t>(totalTwComplex) * 2]();
    prev = 1;
    int64_t len = 1;
    int64_t off = 0;
    for (int32_t radix : radices) {
        len *= radix;
        for (int32_t p = 0; p < radix; ++p) {
            for (int64_t j = 0; j < prev; ++j) {
                double angle = sign * K_2PI * p * j / len;
                host[(off + p * prev + j) * 2] = static_cast<float>(std::cos(angle));
                host[(off + p * prev + j) * 2 + 1] = static_cast<float>(std::sin(angle));
            }
        }
        off += radix * prev;
        prev = len;
    }

    AsdSip::FFTensor *tensor = new AsdSip::FFTensor;
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND,
        {totalTwComplex * 2}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = static_cast<size_t>(totalTwComplex) * 2 * sizeof(float);
    return tensor;
}

AsdSip::FFTensor *BuildMixedDftTensor(const std::vector<int32_t> &radices, bool forward)
{
    double sign = forward ? -1.0 : 1.0;
    int64_t totalDftComplex = 0;
    for (int32_t radix : radices) {
        totalDftComplex += radix * radix;
    }

    float *host = new float[static_cast<size_t>(totalDftComplex) * 2]();
    int64_t off = 0;
    for (int32_t radix : radices) {
        for (int32_t q = 0; q < radix; ++q) {
            for (int32_t p = 0; p < radix; ++p) {
                double angle = sign * K_2PI * p * q / radix;
                host[(off + q * radix + p) * 2] = static_cast<float>(std::cos(angle));
                host[(off + q * radix + p) * 2 + 1] = static_cast<float>(std::sin(angle));
            }
        }
        off += radix * radix;
    }

    AsdSip::FFTensor *tensor = new AsdSip::FFTensor;
    tensor->desc = {Mki::TensorDType::TENSOR_DTYPE_FLOAT, Mki::TensorFormat::TENSOR_FORMAT_ND,
        {totalDftComplex * 2}, {}, 0};
    tensor->hostData = host;
    tensor->dataSize = static_cast<size_t>(totalDftComplex) * 2 * sizeof(float);
    return tensor;
}

void FillRadix2StageTiling(FftC2CArch35StageTilingData &stageTiling, int64_t stageBatch, int64_t fftLen,
    int32_t stage, int32_t outer, int32_t isInverse, bool isLast)
{
    int64_t nHalf = fftLen >> 1;
    int32_t len = 1 << (stage + 1);
    int32_t half = len >> 1;

    stageTiling = {};
    stageTiling.batch = stageBatch;
    stageTiling.n = fftLen;
    stageTiling.totalButterflies = stageBatch * nHalf;
    stageTiling.nHalf = static_cast<int32_t>(nHalf);
    stageTiling.len = len;
    stageTiling.half = half;
    stageTiling.logNhalf = Log2PowerOfTwo(nHalf);
    stageTiling.logHalf = stage;
    stageTiling.twOffset = half - 1;
    stageTiling.outer = outer;
    stageTiling.isInverse = isInverse;
    stageTiling.scaleOut = 0;
    stageTiling.transpose = isLast ? 1 : 0;
}

void FillMixed1DPassTiling(FftAllMixTilingData &tiling, int64_t stageBatch, int64_t fftLen, int32_t radixLen,
    int32_t isInverse, int32_t outer, int32_t transpose)
{
    tiling = {};
    tiling.batchSize = stageBatch;
    tiling.fftN = fftLen;
    tiling.radixListLen = radixLen;
    tiling.isInverse = isInverse;
    tiling.outer = outer;
    tiling.scaleOut = 0;
    tiling.transpose = transpose;

    int64_t singleBufferSize = stageBatch * fftLen * 2 * static_cast<int64_t>(sizeof(float));
    tiling.workspaceOffsets[0] = 0;
    tiling.workspaceOffsets[1] = singleBufferSize;
}

} // namespace

size_t FftC2C2DCoreArch35::EstimateWorkspaceSize()
{
    int64_t singleBufferSize = static_cast<int64_t>(problemDesc.batch) * fftN * fftM * 2 * sizeof(float);
    bool hasMixedPass = ShouldUseMixedRadixKernel(fftM, static_cast<int64_t>(problemDesc.batch) * fftN) ||
        ShouldUseMixedRadixKernel(fftN, static_cast<int64_t>(problemDesc.batch) * fftM);
    int64_t workspaceSize = hasMixedPass ? 3 * singleBufferSize : 2 * singleBufferSize;
    ASDSIP_LOG(INFO) << "ASCEND_950 FftC2C2DCoreArch35 workspace size: " << workspaceSize;
    return static_cast<size_t>(workspaceSize);
}

void FftC2C2DCoreArch35::Run(void *input, void *output, void *stream, workspace::Workspace &workspace)
{
    int64_t batch = static_cast<int64_t>(problemDesc.batch);
    int64_t fullComplexFloats = batch * fftN * fftM * 2;
    bool hasMixedPass = useMixedM || useMixedN;
    int64_t workspaceBuffers = hasMixedPass ? 3 : 2;
    int64_t workspaceBytes = workspaceBuffers * fullComplexFloats * static_cast<int64_t>(sizeof(float));
    float *intermediate = static_cast<float *>(workspace.allocate(static_cast<size_t>(workspaceBytes)));
    float *scratch = intermediate + fullComplexFloats;

    runInfo.SetScratchDeviceAddr(reinterpret_cast<uint8_t *>(hasMixedPass ? scratch : intermediate));
    runInfo.SetStream(stream);

    int32_t isInverse = 1 - static_cast<int32_t>(problemDesc.forward);
    auto launchMixedPass = [&](std::unique_ptr<Mki::Kernel> &passKernel, const std::vector<uint8_t *> &tilingAddrs,
                               void *stageSrc, void *stageDst,
                               const std::shared_ptr<AsdSip::FFTensor> &twTensor,
                               const std::shared_ptr<AsdSip::FFTensor> &radixTensor,
                               const std::shared_ptr<AsdSip::FFTensor> &dftTensor,
                               const FftAllMixTilingData &passTiling) -> bool {
        if (passKernel == nullptr || tilingAddrs.empty()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 mixed-radix pass kernel or tiling buffer is not initialized.";
            return false;
        }

        uint8_t *tilingDeviceAddr = tilingAddrs[0];
        int st = MkiRtMemCopy(tilingDeviceAddr, sizeof(passTiling), &passTiling, sizeof(passTiling),
                              MKIRT_MEMCOPY_HOST_TO_DEVICE);
        if (st != MKIRT_SUCCESS) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 copy mixed pass tiling failed.";
            return false;
        }

        OpParam::FftC2CArch35 passParam = {passTiling.fftN, passTiling.batchSize, passTiling.radixListLen,
                                           passTiling.isInverse, true};

        Tensor passIn;
        Tensor passOut;
        passIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {passTiling.batchSize, passTiling.fftN}, {}, 0};
        passIn.dataSize = static_cast<size_t>(passTiling.batchSize) * static_cast<size_t>(passTiling.fftN) *
                          GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        passIn.data = stageSrc;
        passOut.desc = passIn.desc;
        passOut.dataSize = passIn.dataSize;
        passOut.data = stageDst;

        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(passParam);
        passLaunchParam.AddInTensor(passIn);
        passLaunchParam.AddInTensor(*dftTensor);
        passLaunchParam.AddInTensor(*twTensor);
        passLaunchParam.AddInTensor(*radixTensor);
        passLaunchParam.AddOutTensor(passOut);
        runInfo.SetTilingDeviceAddr(tilingDeviceAddr);

        auto status = passKernel->Run(passLaunchParam, runInfo);
        if (!status.Ok()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 mixed pass launch failed.";
            return false;
        }
        return true;
    };

    auto launchRadix2Stage = [&](std::unique_ptr<Mki::Kernel> &passKernel, uint8_t *stageTilingDeviceAddr,
                                 void *stageSrc, void *stageDst,
                                 const std::shared_ptr<AsdSip::FFTensor> &twTensor,
                                 const std::shared_ptr<AsdSip::FFTensor> &radixTensor,
                                 const std::shared_ptr<AsdSip::FFTensor> &dftTensor,
                                 int64_t radixListLen,
                                 const FftC2CArch35StageTilingData &stageTiling) -> bool {
        if (passKernel == nullptr || stageTilingDeviceAddr == nullptr) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 radix-2 pass kernel or tiling buffer is not initialized.";
            return false;
        }

        int st = MkiRtMemCopy(stageTilingDeviceAddr, sizeof(stageTiling), &stageTiling, sizeof(stageTiling),
                              MKIRT_MEMCOPY_HOST_TO_DEVICE);
        if (st != MKIRT_SUCCESS) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 copy stage tiling failed.";
            return false;
        }

        OpParam::FftC2CArch35 passParam = {stageTiling.n, stageTiling.batch, radixListLen, stageTiling.isInverse,
                                           false};

        Tensor passIn;
        Tensor passOut;
        passIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {stageTiling.batch, stageTiling.n}, {}, 0};
        passIn.dataSize = static_cast<size_t>(stageTiling.batch) * static_cast<size_t>(stageTiling.n) *
                          GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        passIn.data = stageSrc;
        passOut.desc = passIn.desc;
        passOut.dataSize = passIn.dataSize;
        passOut.data = stageDst;

        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(passParam);
        passLaunchParam.AddInTensor(passIn);
        passLaunchParam.AddInTensor(*dftTensor);
        passLaunchParam.AddInTensor(*twTensor);
        passLaunchParam.AddInTensor(*radixTensor);
        passLaunchParam.AddOutTensor(passOut);
        runInfo.SetTilingDeviceAddr(stageTilingDeviceAddr);

        auto status = passKernel->Run(passLaunchParam, runInfo);
        if (!status.Ok()) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 radix-2 stage launch failed.";
            return false;
        }
        return true;
    };

    if (useMixedM) {
        FftAllMixTilingData passM = {};
        FillMixed1DPassTiling(passM, batch * fftN, fftM, static_cast<int32_t>(radixListMHost.size()), isInverse,
            static_cast<int32_t>(fftN), 1);
        if (!launchMixedPass(kernelM, tilingMDeviceAddrs, input, intermediate, twMatrixM, radixListM, dftMatrixM,
                passM)) {
            workspace.recycleLast();
            return;
        }
    } else {
        int32_t cntStagesM = Log2PowerOfTwo(fftM);
        if (tilingMDeviceAddrs.size() < static_cast<size_t>(cntStagesM)) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 M radix-2 stage tiling buffers are not initialized.";
            workspace.recycleLast();
            return;
        }
        bool firstMDstScratch = (cntStagesM % 2 == 0);
        for (int32_t stage = 0; stage < cntStagesM; ++stage) {
            bool isLast = (stage == cntStagesM - 1);
            void *stageSrc = nullptr;
            void *stageDst = nullptr;
            if (stage == 0) {
                stageSrc = input;
            } else {
                bool prevDstScratch = firstMDstScratch ? (((stage - 1) & 1) == 0) : (((stage - 1) & 1) != 0);
                stageSrc = prevDstScratch ? static_cast<void *>(scratch) : static_cast<void *>(intermediate);
            }
            if (isLast) {
                stageDst = intermediate;
            } else {
                bool dstScratch = firstMDstScratch ? ((stage & 1) == 0) : ((stage & 1) != 0);
                stageDst = dstScratch ? static_cast<void *>(scratch) : static_cast<void *>(intermediate);
            }

            FftC2CArch35StageTilingData stageTiling;
            FillRadix2StageTiling(stageTiling, batch * fftN, fftM, stage, static_cast<int32_t>(fftN), isInverse,
                isLast);
            if (!launchRadix2Stage(kernelM, tilingMDeviceAddrs[stage], stageSrc, stageDst, twMatrixM, radixListM,
                    dftMatrixM, static_cast<int64_t>(radixListMHost.size()), stageTiling)) {
                workspace.recycleLast();
                return;
            }
        }
    }

    if (useMixedN) {
        FftAllMixTilingData passN = {};
        FillMixed1DPassTiling(passN, batch * fftM, fftN, static_cast<int32_t>(radixListNHost.size()), isInverse,
            static_cast<int32_t>(fftM), 1);
        if (!launchMixedPass(kernelN, tilingNDeviceAddrs, intermediate, output, twMatrixN, radixListN, dftMatrixN,
                passN)) {
            workspace.recycleLast();
            return;
        }
    } else {
        int32_t cntStagesN = Log2PowerOfTwo(fftN);
        if (tilingNDeviceAddrs.size() < static_cast<size_t>(cntStagesN)) {
            ASDSIP_LOG(ERROR) << "FftC2C2DCoreArch35 N radix-2 stage tiling buffers are not initialized.";
            workspace.recycleLast();
            return;
        }
        void *currentN = intermediate;
        for (int32_t stage = 0; stage < cntStagesN; ++stage) {
            bool isLast = (stage == cntStagesN - 1);
            void *stageDst = isLast ? output : ((stage & 1) == 0 ? static_cast<void *>(scratch) :
                static_cast<void *>(intermediate));

            FftC2CArch35StageTilingData stageTiling;
            FillRadix2StageTiling(stageTiling, batch * fftM, fftN, stage, static_cast<int32_t>(fftM), isInverse,
                isLast);
            if (!launchRadix2Stage(kernelN, tilingNDeviceAddrs[stage], currentN, stageDst, twMatrixN, radixListN,
                    dftMatrixN, static_cast<int64_t>(radixListNHost.size()), stageTiling)) {
                workspace.recycleLast();
                return;
            }
            currentN = stageDst;
        }
    }

    workspace.recycleLast();
    ASDSIP_LOG(INFO) << "ASCEND_950 FftC2C2DCoreArch35 run success.";
}

void FftC2C2DCoreArch35::DestroyInDevice()
{
    FreeDeviceAddrs(tilingMDeviceAddrs);
    FreeDeviceAddrs(tilingNDeviceAddrs);
}

AspbStatus FftC2C2DCoreArch35::BuildFftPlan()
{
    radixListNHost = FactorizeMixed(fftN);
    radixListMHost = FactorizeMixed(fftM);
    useMixedM = ShouldUseMixedRadixKernel(fftM, static_cast<int64_t>(problemDesc.batch) * fftN);
    useMixedN = ShouldUseMixedRadixKernel(fftN, static_cast<int64_t>(problemDesc.batch) * fftM);

    auto radixListMLocal = radixListMHost;
    auto radixListNLocal = radixListNHost;

    AsdSip::CoeffKey radixMKey = {coreType, useMixedM ? 211U : 202U, {fftM}, problemDesc.forward};
    AsdSip::CoeffKey radixNKey = {coreType, useMixedN ? 210U : 203U, {fftN}, problemDesc.forward};
    AsdSip::CoeffKey twMKey = {coreType, useMixedM ? 213U : 200U, {fftM}, problemDesc.forward};
    AsdSip::CoeffKey twNKey = {coreType, useMixedN ? 212U : 201U, {fftN}, problemDesc.forward};
    AsdSip::CoeffKey dftMKey = {coreType, useMixedM ? 215U : 204U, {fftM}, problemDesc.forward};
    AsdSip::CoeffKey dftNKey = {coreType, useMixedN ? 214U : 205U, {fftN}, problemDesc.forward};

    radixListM = FFTensorCache::getCoeff(radixMKey, [=]() -> AsdSip::FFTensor * {
        return BuildRadixListTensor(radixListMLocal);
    });
    radixListN = FFTensorCache::getCoeff(radixNKey, [=]() -> AsdSip::FFTensor * {
        return BuildRadixListTensor(radixListNLocal);
    });
    twMatrixM = FFTensorCache::getCoeff(twMKey, [=]() -> AsdSip::FFTensor * {
        return useMixedM ? BuildMixedTwiddleTensor(radixListMLocal, problemDesc.forward) :
            BuildRadix2TwiddleTensor(fftM, problemDesc.forward);
    });
    twMatrixN = FFTensorCache::getCoeff(twNKey, [=]() -> AsdSip::FFTensor * {
        return useMixedN ? BuildMixedTwiddleTensor(radixListNLocal, problemDesc.forward) :
            BuildRadix2TwiddleTensor(fftN, problemDesc.forward);
    });
    dftMatrixM = FFTensorCache::getCoeff(dftMKey, [=]() -> AsdSip::FFTensor * {
        return useMixedM ? BuildMixedDftTensor(radixListMLocal, problemDesc.forward) : BuildRadix2DftTensor();
    });
    dftMatrixN = FFTensorCache::getCoeff(dftNKey, [=]() -> AsdSip::FFTensor * {
        return useMixedN ? BuildMixedDftTensor(radixListNLocal, problemDesc.forward) : BuildRadix2DftTensor();
    });

    ASDSIP_LOG(INFO) << "FftC2C2DCoreArch35 BuildFftPlan success. fftN=" << fftN << ", fftM=" << fftM
                     << ", useMixedN=" << useMixedN << ", useMixedM=" << useMixedM;
    return AsdSip::ErrorType::ACL_SUCCESS;
}

bool FftC2C2DCoreArch35::PreAllocateInDevice()
{
    if (BuildFftPlan() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    if (InitTactic() != AsdSip::ErrorType::ACL_SUCCESS) {
        return false;
    }

    return true;
}

AspbStatus FftC2C2DCoreArch35::InitTactic()
{
    int32_t isInverse = 1 - static_cast<int32_t>(problemDesc.forward);

    auto initPass = [&](bool useMixed, int64_t fftLen, int64_t passBatch, int64_t radixListLen,
                        const std::shared_ptr<AsdSip::FFTensor> &dftTensor,
                        const std::shared_ptr<AsdSip::FFTensor> &twTensor,
                        const std::shared_ptr<AsdSip::FFTensor> &radixTensor,
                        std::unique_ptr<Mki::Kernel> &passKernel,
                        std::vector<uint8_t *> &tilingAddrs) -> AsdSip::AspbStatus {
        Tensor tensorIn;
        Tensor tensorOut;
        tensorIn.desc = {TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {passBatch, fftLen}, {}, 0};
        tensorIn.dataSize = static_cast<size_t>(passBatch) * static_cast<size_t>(fftLen) *
                            GetTensorElementSize(Mki::TensorDType::TENSOR_DTYPE_COMPLEX64);
        tensorOut.desc = tensorIn.desc;
        tensorOut.dataSize = tensorIn.dataSize;

        OpParam::FftC2CArch35 param = {fftLen, passBatch, radixListLen, isInverse, useMixed};
        Mki::LaunchParam passLaunchParam;
        passLaunchParam.SetParam(param);
        passLaunchParam.AddInTensor(tensorIn);
        passLaunchParam.AddInTensor(*dftTensor);
        passLaunchParam.AddInTensor(*twTensor);
        passLaunchParam.AddInTensor(*radixTensor);
        passLaunchParam.AddOutTensor(tensorOut);

        Operation *op = Ops::Instance().GetOperationByName(opName);
        if (op == nullptr) {
            return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
        }
        op->InferShape(passLaunchParam);
        passKernel = std::unique_ptr<Mki::Kernel>(op->GetBestKernel(passLaunchParam));
        ASDSIP_ECHECK(passKernel != nullptr, "Get best kernel failed", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        passKernel->SetLaunchWithTiling(false);
        uint32_t launchBufferSize = passKernel->GetTilingSize(passLaunchParam);
        ASDSIP_ECHECK(launchBufferSize != 0, "empty tiling size", AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR);

        std::vector<uint8_t> hostLaunchBuffer(launchBufferSize);
        passKernel->SetTilingHostAddr(hostLaunchBuffer.data(), launchBufferSize);
        passKernel->Init(passLaunchParam);

        int32_t bufferCount = useMixed ? 1 : Log2PowerOfTwo(fftLen);
        FreeDeviceAddrs(tilingAddrs);
        tilingAddrs.reserve(static_cast<size_t>(bufferCount));
        for (int32_t buffer = 0; buffer < bufferCount; ++buffer) {
            void *tempDevicePtr = nullptr;
            int st = MkiRtMemMallocDevice(&tempDevicePtr, launchBufferSize, MKIRT_MEM_DEFAULT);
            if (st != MKIRT_SUCCESS) {
                FreeDeviceAddrs(tilingAddrs);
                ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "malloc pass tiling device memory fail";
                return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
            }

            uint8_t *deviceLaunchBuffer = static_cast<uint8_t *>(tempDevicePtr);
            st = MkiRtMemCopy(deviceLaunchBuffer, launchBufferSize, hostLaunchBuffer.data(), launchBufferSize,
                              MKIRT_MEMCOPY_HOST_TO_DEVICE);
            if (st != MKIRT_SUCCESS) {
                MkiRtMemFreeDevice(deviceLaunchBuffer);
                FreeDeviceAddrs(tilingAddrs);
                ASDSIP_ELOG(AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR) << "copy pass tiling to device fail";
                return AsdSip::ErrorType::ACL_ERROR_INTERNAL_ERROR;
            }
            tilingAddrs.push_back(deviceLaunchBuffer);
        }
        return AsdSip::ErrorType::ACL_SUCCESS;
    };

    auto status = initPass(useMixedM, fftM, static_cast<int64_t>(problemDesc.batch) * fftN,
        static_cast<int64_t>(radixListMHost.size()), dftMatrixM, twMatrixM, radixListM, kernelM, tilingMDeviceAddrs);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        return status;
    }

    status = initPass(useMixedN, fftN, static_cast<int64_t>(problemDesc.batch) * fftM,
        static_cast<int64_t>(radixListNHost.size()), dftMatrixN, twMatrixN, radixListN, kernelN, tilingNDeviceAddrs);
    if (status != AsdSip::ErrorType::ACL_SUCCESS) {
        return status;
    }

    ASDSIP_LOG(INFO) << "FftC2C2DCoreArch35 init tactic success.";
    return AsdSip::ErrorType::ACL_SUCCESS;
}
