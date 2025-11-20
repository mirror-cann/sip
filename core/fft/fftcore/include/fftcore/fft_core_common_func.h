/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTCORE_COMMONFUNC__
#define __FFTCORE_COMMONFUNC__

#include <cmath>
#include <vector>
#include <memory>
#include "ops.h"
#include "utils/ops_base.h"
#include "utils/aspb_status.h"

#include "fftcore/fft_core_base.h"

constexpr int64_t K_SIZE_OF_COMPLEX64 = sizeof(float) * 2;

AsdSip::AspbStatus InitWMatrixCommon(FFTCoreType coreType, std::vector<int64_t> &radixVec, int64_t n, bool forward,
                                     std::shared_ptr<AsdSip::FFTensor> &wMatrix);

AsdSip::AspbStatus InitTMatrixCommon(FFTCoreType coreType, std::vector<int64_t> &radixVec, int64_t n,
                                     std::shared_ptr<AsdSip::FFTensor> &tMatrix);

AsdSip::AspbStatus InitSMatrixCommon(FFTCoreType coreType, std::vector<int64_t> &radixVec, int64_t n, bool forward,
                                     std::shared_ptr<AsdSip::FFTensor> &sMatrix);

void InitMixRadixLong(int64_t fftN, std::vector<int64_t> &radixVecRef);

void InitMixRadixShort(int64_t fftN, std::vector<int64_t> &radixVecRef);

AsdSip::AspbStatus InitMixRadixList(FFTCoreType coreType, int64_t fftN, bool forward,
                                    const std::vector<int64_t> &radixVecRef,
                                    std::shared_ptr<AsdSip::FFTensor> &radixList);

AsdSip::AspbStatus InitMixRadixTwiddleMatrix(FFTCoreType coreType, int64_t fftN, int64_t batchSize, bool forward,
                                             const std::vector<int64_t> &radixVecRef,
                                             std::shared_ptr<AsdSip::FFTensor> &twiddleMatrixArray);

AsdSip::AspbStatus InitMixRadixTWMatrix(FFTCoreType coreType, int64_t fftN, bool forward,
                                        const std::vector<int64_t> &radixVecRef,
                                        std::shared_ptr<AsdSip::FFTensor> &twMatrixArray);

AsdSip::AspbStatus InitDftTwiddleMatrix(FFTCoreType coreType, int64_t fftN, bool forward,
                                        std::shared_ptr<AsdSip::FFTensor> &rotationMatrix);

AsdSip::AspbStatus InitInputOutputIndex(FFTCoreType coreType, int64_t fftN, int parity,
                                        std::shared_ptr<AsdSip::FFTensor> &inputIndex,
                                        std::shared_ptr<AsdSip::FFTensor> &outputIndex);

AsdSip::AspbStatus InitABTable(FFTCoreType coreType, int64_t fftN, bool forward, int parity,
                               std::shared_ptr<AsdSip::FFTensor> &aTable, std::shared_ptr<AsdSip::FFTensor> &bTable);

void InitMixRadixParam(int64_t parity, int64_t fftN, int64_t batchSize, const std::vector<int64_t> &radixVecRef,
                       int64_t &workspaceInputSize, int64_t &workspaceOutputSize, int64_t &workspaceSyncSize,
                       int64_t &workspaceC2cOutputSize, int64_t &workspaceAuxilSize);

void InitAiVSplitWay(FFTCoreType coreType, int64_t fftN, const std::vector<int64_t> &radixVecRef, int32_t &aivSplitWay);

AsdSip::AspbStatus InitPQMatrix(FFTCoreType coreType, int64_t fftN, bool forward, bool isP,
                                std::shared_ptr<AsdSip::FFTensor> &pqMatrix);

void GenWMatrixForwardForMultiLen(int64_t stepLen, int64_t *radixListPtr, int64_t fftN, float *nowDftMatrixArrayHost);

void GenWMatrixForwardForOneLen(int64_t stepLen, int64_t *radixListPtr, int64_t fftN, float *nowDftMatrixArrayHost);

void GenWMatrixInverseForMultiLen(int64_t stepLen, int64_t *radixListPtr, int64_t fftN,
                                  float *nowDftMatrixArrayInverseHost);

void GenWMatrixInverseForOneLen(int64_t stepLen, int64_t *radixListPtr, int64_t fftN,
                                float *nowDftMatrixArrayInverseHost);

#endif