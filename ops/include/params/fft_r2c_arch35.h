/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_FFT_R2C_ARCH35_H
#define ASDSIP_PARAMS_FFT_R2C_ARCH35_H

#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {

/**
 * Parameter struct for FftR2C Arch35 (Ascend 950) operation.
 *
 * This is a streamlined parameter for the mixed-radix FFT R2C kernel
 * that only supports prime radices 2, 3, 5, 7.
 *
 * Fields:
 *   fftN         - Full real signal length N (input length)
 *   batchSize    - Number of batched FFTs
 *   radixListLen - Number of radix decomposition stages (length of plan for N/2 FFT)
 *   isInverse    - 0 for R2C (forward), 1 for C2R (inverse)
 *   isOddN       - 1 if N is odd (use full N-point FFT), 0 if N is even (use N/2 Pack optimization)
 */
struct FftR2CArch35 {
    int64_t fftN;
    int64_t batchSize;
    int64_t radixListLen;
    int32_t isInverse;
    int32_t isOddN;  // 新增：标记是否为奇数 N

    bool operator==(const FftR2CArch35 &other) const
    {
        return this->fftN == other.fftN && this->batchSize == other.batchSize &&
               this->radixListLen == other.radixListLen && this->isInverse == other.isInverse &&
               this->isOddN == other.isOddN;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: FftR2CArch35"
           << ", fftN:" << fftN << ", batchSize:" << batchSize
           << ", radixListLen:" << radixListLen << ", isInverse:" << isInverse
           << ", isOddN:" << isOddN;
        return ss.str();
    }
};

}  // namespace OpParam
}  // namespace AsdSip

#endif