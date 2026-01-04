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
constexpr int64_t K_SIZE_OF_FLOAT = sizeof(float);

constexpr int W_MARK = 1;
constexpr int T_MARK = 2;
constexpr int S_MARK = 3;
constexpr int DD_P_MARK = 0;
constexpr int DD_Q_MARK = 1;
constexpr int DFT_TWIDDLE_MARK = 0;
constexpr int INPUT_MARK = 1;
constexpr int OUTPUT_MARK = 2;
constexpr int A_MARK = 3;
constexpr int B_MARK = 4;
constexpr int MIX_RADIX_LIST_MARK = 5;
constexpr int MIX_FORWARD_MARK = 6;
constexpr int MIX_BACKWARD_MARK = 7;
constexpr int MIX_TW_MARK = 8;
constexpr int TWO_MUL = 2;
constexpr int RADIX_8K = 8192;
constexpr int RADIX_16K = 16384;
constexpr int RADIX_ADDR_START = 128;
constexpr int RADIX_ADDR_END = 2;
constexpr int32_t RADIX_LEN_19683 = 19683;
constexpr int32_t RADIX_LEN_243 = 243;
constexpr int32_t RADIX_LEN_729 = 729;
constexpr int32_t RADIX_LEN_59049 = 59049;
constexpr int32_t RADIX_LEN_177147 = 177147;
constexpr int32_t RADIX_LEN_1594323 = 1594323;
constexpr int32_t RADIX_LEN_129140163 = 129140163;
constexpr int32_t RADIX_LEN_387420489 = 387420489;
constexpr int32_t RADIX_LEN_625 = 625;
constexpr int32_t RADIX_LEN_3125 = 3125;
constexpr int32_t RADIX_LEN_15625 = 15625;
constexpr int32_t RADIX_LEN_78125 = 78125;
constexpr int32_t RADIX_LEN_390625 = 390625;
constexpr int32_t RADIX_LEN_2401 = 2401;
constexpr int32_t RADIX_LEN_16807 = 16807;
constexpr int32_t RADIX_LEN_117649 = 117649;
constexpr int32_t RADIX_LEN_76800 = 76800;
constexpr int32_t RADIX_LEN_EVEN_NUM = 2;
constexpr int32_t RADIX_LEN_THREE_NUM = 3;
constexpr int32_t RADIX_LEN_FIVE_NUM = 5;
constexpr int32_t RADIX_LEN_SEVEN_NUM = 7;
constexpr int32_t RADIX_LEN_FOUR_NUM = 4;
constexpr int32_t ROUND_16 = 16;
constexpr int32_t ROUND_32 = 32;
constexpr int32_t AUXILSIZE = 64;
constexpr int32_t VECVTRANS_LEN = 88;
constexpr int32_t RADIX_LIST_LEN = 45;
constexpr int32_t AIVSPLITWAY_TWO = 2;
constexpr int32_t AIVSPLITWAY_THREE = 3;


AsdSip::AspbStatus InitWMatrixCommon(FFTCoreType coreType, std::vector<int64_t> &radixVec, int64_t n, bool forward,
                                     std::shared_ptr<AsdSip::FFTensor> &wMatrix);

AsdSip::AspbStatus InitTMatrixCommon(FFTCoreType coreType, std::vector<int64_t> &radixVec, int64_t n,
                                     std::shared_ptr<AsdSip::FFTensor> &tMatrix);

AsdSip::AspbStatus InitSMatrixCommon(FFTCoreType coreType, std::vector<int64_t> &radixVec, int64_t n, bool forward,
                                     std::shared_ptr<AsdSip::FFTensor> &sMatrix);

void InitMixRadixLong(int64_t fftN, std::vector<int64_t> &radixVecRef);

void InitMixRadixShort(int64_t fftN, std::vector<int64_t> &radixVecRef);

AsdSip::AspbStatus InitMixRadixList(FFTCoreType coreType, int64_t fftN, bool forward,
                                    std::vector<int64_t> &radixVecRef,
                                    std::shared_ptr<AsdSip::FFTensor> &radixList);

AsdSip::AspbStatus InitMixRadixTwiddleMatrix(FFTCoreType coreType, int64_t fftN, int64_t batchSize, bool forward,
                                             std::vector<int64_t> &radixVecRef,
                                             std::shared_ptr<AsdSip::FFTensor> &twiddleMatrixArray);

AsdSip::AspbStatus InitMixRadixTWMatrix(FFTCoreType coreType, int64_t fftN, bool forward,
                                        std::vector<int64_t> &radixVecRef,
                                        std::shared_ptr<AsdSip::FFTensor> &twMatrixArray);

AsdSip::AspbStatus InitDftTwiddleMatrix(FFTCoreType coreType, int64_t fftN, bool forward,
                                        std::shared_ptr<AsdSip::FFTensor> &rotationMatrix);

AsdSip::AspbStatus InitInputOutputIndex(FFTCoreType coreType, int64_t fftN, int parity,
                                        std::shared_ptr<AsdSip::FFTensor> &inputIndex,
                                        std::shared_ptr<AsdSip::FFTensor> &outputIndex);

AsdSip::AspbStatus InitABTable(FFTCoreType coreType, int64_t fftN, bool forward, int parity,
                               std::shared_ptr<AsdSip::FFTensor> &aTable, std::shared_ptr<AsdSip::FFTensor> &bTable);

void InitMixRadixParam(int64_t parity, int64_t fftN, int64_t batchSize, std::vector<int64_t> &radixVecRef,
                       int64_t &workspaceInputSize, int64_t &workspaceOutputSize, int64_t &workspaceSyncSize,
                       int64_t &workspaceC2cOutputSize, int64_t &workspaceAuxilSize);

void InitAiVSplitWay(FFTCoreType coreType, int64_t fftN, std::vector<int64_t> &radixVecRef, int32_t &aivSplitWay);

AsdSip::AspbStatus InitPQMatrix(FFTCoreType coreType, int64_t fftN, bool forward, bool isP,
                                std::shared_ptr<AsdSip::FFTensor> &pqMatrix);

void GenWMatrixForwardForMultiLen(int64_t stepLen, int64_t *radixListPtr, int64_t fftN, float *nowDftMatrixArrayHost);

void GenWMatrixForwardForOneLen(int64_t stepLen, int64_t *radixListPtr, int64_t fftN, float *nowDftMatrixArrayHost);

void GenWMatrixInverseForMultiLen(int64_t stepLen, int64_t *radixListPtr, int64_t fftN,
                                  float *nowDftMatrixArrayInverseHost);

void GenWMatrixInverseForOneLen(int64_t stepLen, int64_t *radixListPtr, int64_t fftN,
                                float *nowDftMatrixArrayInverseHost);

#endif