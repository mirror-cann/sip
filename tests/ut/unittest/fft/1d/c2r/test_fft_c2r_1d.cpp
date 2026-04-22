/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include <gtest/gtest.h>
#include <filesystem>
#include <cmath>
#include <complex>
#include <fstream>
#include <random>
#include "test_common.h"
#include "mki/utils/status/status.h"
#include "mki/utils/log/log.h"
#include "test_util/float_util.h"
#include "test_util/util.cpp"
#include "ops.h"
#include "fft_api.h"
#include "acl/acl.h"
#include "aclnn/acl_meta.h"
using namespace AsdSip;
using namespace Mki;

namespace {
constexpr float ATOL = 0.001;
constexpr float RTOL = 0.001;

std::string GetC2ROutputDirectory() {
    const char* current_dir_env = std::getenv("CURRENT_DIR");
    if (current_dir_env) {
        return std::string(current_dir_env);
    } else {
        return std::string("get env failed.");
    }
}
} // namespace

/**
 * Test FFT C2R 1D with batch=1, nfft=16
 */
TEST(TestFftC2R1d, TestC2RCase0)
{
    // Get data directory path
    std::string originPath = GetC2ROutputDirectory() + "/tests/ut/unittest/fft/1d/c2r/c2r_data";
    std::string destPath = GetC2ROutputDirectory() + "/build/tests/ut/unittest/fft/1d/c2r";
    std::string mkdir_cmd = "mkdir -p " + destPath;
    std::string copy_cmd = "cp -rf " + originPath + " " + destPath;
    std::string chmod_cmd = "chmod -R 755 " + destPath + "/c2r_data/";
    system(mkdir_cmd.c_str());
    system(copy_cmd.c_str());
    system(chmod_cmd.c_str());

    int deviceId = 0;
    int64_t batch = 1;
    int64_t nfft = 16;
    int64_t inSignal = nfft / 2 + 1;
    int64_t outSignal = nfft;

    MkiRtStream stream = OpTestInit(deviceId);

    // Create input tensor descriptor (complex64)
    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, inSignal}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch, outSignal}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    // Save input data to bin file
    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) + ".bin";
        if (SaveTensorToBin(context.inTensors[i], destPath + "/c2r_data/" + filename)) {
            std::cout << "Tensor saved successfully: " << filename << std::endl;
        } else {
            std::cerr << "Failed to save tensor: " << filename << std::endl;
        }
    }

    // Generate golden data via Python
    std::string gen_data_cmd = "cd " + destPath + "/c2r_data/" + " && python3 gen_data.py --batch " + std::to_string(batch) + " --nfft " + std::to_string(nfft);
    system(gen_data_cmd.c_str());

    // Create aclTensor for FFT API
    aclTensor *aclInput = nullptr;
    aclTensor *aclOutput = nullptr;
    void *inputDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    // Prepare host data
    std::vector<std::complex<float>> inputHostData(batch * inSignal);
    std::complex<float>* inputDataPtr = static_cast<std::complex<float>*>(inputTensor.hostData);
    for (int64_t i = 0; i < batch * inSignal; i++) {
        inputHostData[i] = inputDataPtr[i];
    }

    std::vector<float> outputHostData(batch * outSignal, 0.0f);
    std::vector<int64_t> inShape = {batch, inSignal};
    std::vector<int64_t> outShape = {batch, outSignal};

    // Create aclTensors
    auto ret = CreateAclTensor(inputHostData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputHostData, outShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &aclOutput);
    ASSERT_EQ(ret, 0);

    // Execute FFT C2R
    asdFftHandle handle;
    AspbStatus fftStatus = asdFftCreate(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftMakePlan1D(handle, nfft, asdFftType::ASCEND_FFT_C2R, asdFftDirection::ASCEND_FFT_FORWARD, batch);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    fftStatus = asdFftGetWorkspaceSize(handle, workSize);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    void *workspaceAddr = nullptr;
    if (workSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(ret, ::ACL_SUCCESS);
        fftStatus = asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
        ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);
    }

    // Convert MkiRtStream to aclrtStream
    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    fftStatus = asdFftSetStream(handle, aclStream);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftExecC2R(handle, aclInput, aclOutput);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftSynchronize(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    asdFftDestroy(handle);

    // Copy output back to host from aclOutput's device memory
    ret = aclrtMemcpy(outputTensor.hostData,
                      outputTensor.dataSize,
                      outputDeviceAddr,
                      outputTensor.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    
    // Save output to bin file
    std::string outFilename = GetElementDtype(outputTensor.desc.dtype) + "_output_.bin";
    if (SaveOutTensorToBin(outputTensor, destPath + "/c2r_data/" + outFilename)) {
        std::cout << "Output tensor saved successfully" << std::endl;
    } else {
        std::cerr << "Failed to save output tensor" << std::endl;
    }

    // Cleanup aclTensors
    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    OpTestEnd(deviceId, context, stream);

    // Compare results via Python
    std::string cmp_data_cmd = "cd " + destPath + "/c2r_data/" + " && python3 compare_data.py --batch " + std::to_string(batch) + " --nfft " + std::to_string(nfft);
    int res = system(cmp_data_cmd.c_str());
    std::cout << "compare result = " << res << std::endl;
    ASSERT_EQ(res, 0);
}

/**
 * Test FFT C2R 1D with batch=2, nfft=32
 */
TEST(TestFftC2R1d, TestC2RCase1)
{
    std::string originPath = GetC2ROutputDirectory() + "/tests/ut/unittest/fft/1d/c2r/c2r_data";
    std::string destPath = GetC2ROutputDirectory() + "/build/tests/ut/unittest/fft/1d/c2r";
    std::string mkdir_cmd = "mkdir -p " + destPath;
    std::string copy_cmd = "cp -rf " + originPath + " " + destPath;
    std::string chmod_cmd = "chmod -R 755 " + destPath + "/c2r_data/";
    system(mkdir_cmd.c_str());
    system(copy_cmd.c_str());
    system(chmod_cmd.c_str());

    int deviceId = 0;
    int64_t batch = 2;
    int64_t nfft = 32;
    int64_t inSignal = nfft / 2 + 1;
    int64_t outSignal = nfft;

    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, inSignal}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch, outSignal}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) + ".bin";
        SaveTensorToBin(context.inTensors[i], destPath + "/c2r_data/" + filename);
    }

    std::string gen_data_cmd = "cd " + destPath + "/c2r_data/" + " && python3 gen_data.py --batch " + std::to_string(batch) + " --nfft " + std::to_string(nfft);
    system(gen_data_cmd.c_str());

    aclTensor *aclInput = nullptr;
    aclTensor *aclOutput = nullptr;
    void *inputDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    std::vector<std::complex<float>> inputHostData(batch * inSignal);
    std::complex<float>* inputDataPtr = static_cast<std::complex<float>*>(inputTensor.hostData);
    for (int64_t i = 0; i < batch * inSignal; i++) {
        inputHostData[i] = inputDataPtr[i];
    }

    std::vector<float> outputHostData(batch * outSignal, 0.0f);
    std::vector<int64_t> inShape = {batch, inSignal};
    std::vector<int64_t> outShape = {batch, outSignal};

    auto ret = CreateAclTensor(inputHostData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputHostData, outShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &aclOutput);
    ASSERT_EQ(ret, 0);

    asdFftHandle handle;
    AspbStatus fftStatus = asdFftCreate(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftMakePlan1D(handle, nfft, asdFftType::ASCEND_FFT_C2R, asdFftDirection::ASCEND_FFT_FORWARD, batch);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    fftStatus = asdFftGetWorkspaceSize(handle, workSize);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    void *workspaceAddr = nullptr;
    if (workSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(ret, ::ACL_SUCCESS);
        fftStatus = asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
        ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);
    }

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    fftStatus = asdFftSetStream(handle, aclStream);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftExecC2R(handle, aclInput, aclOutput);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftSynchronize(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    asdFftDestroy(handle);

    ret = aclrtMemcpy(outputTensor.hostData,
                      outputTensor.dataSize,
                      outputDeviceAddr,
                      outputTensor.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    
    std::string outFilename = GetElementDtype(outputTensor.desc.dtype) + "_output_.bin";
    SaveOutTensorToBin(outputTensor, destPath + "/c2r_data/" + outFilename);

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    OpTestEnd(deviceId, context, stream);

    std::string cmp_data_cmd = "cd " + destPath + "/c2r_data/" + " && python3 compare_data.py --batch " + std::to_string(batch) + " --nfft " + std::to_string(nfft);
    int res = system(cmp_data_cmd.c_str());
    ASSERT_EQ(res, 0);
}

/**
 * Test FFT C2R 1D with batch=1, nfft=64
 */
TEST(TestFftC2R1d, TestC2RCase2)
{
    std::string originPath = GetC2ROutputDirectory() + "/tests/ut/unittest/fft/1d/c2r/c2r_data";
    std::string destPath = GetC2ROutputDirectory() + "/build/tests/ut/unittest/fft/1d/c2r";
    std::string mkdir_cmd = "mkdir -p " + destPath;
    std::string copy_cmd = "cp -rf " + originPath + " " + destPath;
    std::string chmod_cmd = "chmod -R 755 " + destPath + "/c2r_data/";
    system(mkdir_cmd.c_str());
    system(copy_cmd.c_str());
    system(chmod_cmd.c_str());

    int deviceId = 0;
    int64_t batch = 1;
    int64_t nfft = 64;
    int64_t inSignal = nfft / 2 + 1;
    int64_t outSignal = nfft;

    MkiRtStream stream = OpTestInit(deviceId);

    SVector<TensorDesc> inTensorDescs;
    inTensorDescs.push_back({TENSOR_DTYPE_COMPLEX64, TENSOR_FORMAT_ND, {batch, inSignal}});
    inTensorDescs.push_back({TENSOR_DTYPE_FLOAT, TENSOR_FORMAT_ND, {batch, outSignal}});

    TensorContext context;
    OpTestMallocInTensors(inTensorDescs, context);

    Tensor inputTensor = context.inTensors[0];
    Tensor outputTensor = context.inTensors[1];

    for (size_t i = 0; i < context.inTensors.size(); i++) {
        std::string filename = GetElementDtype(context.inTensors[i].desc.dtype) + "_input_" + std::to_string(i) + ".bin";
        SaveTensorToBin(context.inTensors[i], destPath + "/c2r_data/" + filename);
    }

    std::string gen_data_cmd = "cd " + destPath + "/c2r_data/" + " && python3 gen_data.py --batch " + std::to_string(batch) + " --nfft " + std::to_string(nfft);
    system(gen_data_cmd.c_str());

    aclTensor *aclInput = nullptr;
    aclTensor *aclOutput = nullptr;
    void *inputDeviceAddr = nullptr;
    void *outputDeviceAddr = nullptr;

    std::vector<std::complex<float>> inputHostData(batch * inSignal);
    std::complex<float>* inputDataPtr = static_cast<std::complex<float>*>(inputTensor.hostData);
    for (int64_t i = 0; i < batch * inSignal; i++) {
        inputHostData[i] = inputDataPtr[i];
    }

    std::vector<float> outputHostData(batch * outSignal, 0.0f);
    std::vector<int64_t> inShape = {batch, inSignal};
    std::vector<int64_t> outShape = {batch, outSignal};

    auto ret = CreateAclTensor(inputHostData, inShape, &inputDeviceAddr, aclDataType::ACL_COMPLEX64, &aclInput);
    ASSERT_EQ(ret, 0);
    ret = CreateAclTensor(outputHostData, outShape, &outputDeviceAddr, aclDataType::ACL_FLOAT, &aclOutput);
    ASSERT_EQ(ret, 0);

    asdFftHandle handle;
    AspbStatus fftStatus = asdFftCreate(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftMakePlan1D(handle, nfft, asdFftType::ASCEND_FFT_C2R, asdFftDirection::ASCEND_FFT_FORWARD, batch);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    size_t workSize = 0;
    fftStatus = asdFftGetWorkspaceSize(handle, workSize);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    void *workspaceAddr = nullptr;
    if (workSize > 0) {
        ret = aclrtMalloc(&workspaceAddr, static_cast<int64_t>(workSize), ACL_MEM_MALLOC_HUGE_FIRST);
        ASSERT_EQ(ret, ::ACL_SUCCESS);
        fftStatus = asdFftSetWorkspace(handle, (uint8_t *)workspaceAddr);
        ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);
    }

    aclrtStream aclStream = static_cast<aclrtStream>(stream);
    fftStatus = asdFftSetStream(handle, aclStream);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftExecC2R(handle, aclInput, aclOutput);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    fftStatus = asdFftSynchronize(handle);
    ASSERT_EQ(fftStatus, AsdSip::ErrorType::ACL_SUCCESS);

    asdFftDestroy(handle);

    ret = aclrtMemcpy(outputTensor.hostData,
                      outputTensor.dataSize,
                      outputDeviceAddr,
                      outputTensor.dataSize,
                      ACL_MEMCPY_DEVICE_TO_HOST);
    ASSERT_EQ(ret, ::ACL_SUCCESS);
    
    std::string outFilename = GetElementDtype(outputTensor.desc.dtype) + "_output_.bin";
    SaveOutTensorToBin(outputTensor, destPath + "/c2r_data/" + outFilename);

    aclDestroyTensor(aclInput);
    aclDestroyTensor(aclOutput);
    aclrtFree(inputDeviceAddr);
    aclrtFree(outputDeviceAddr);
    if (workSize > 0) {
        aclrtFree(workspaceAddr);
    }

    OpTestEnd(deviceId, context, stream);

    std::string cmp_data_cmd = "cd " + destPath + "/c2r_data/" + " && python3 compare_data.py --batch " + std::to_string(batch) + " --nfft " + std::to_string(nfft);
    int res = system(cmp_data_cmd.c_str());
    ASSERT_EQ(res, 0);
}