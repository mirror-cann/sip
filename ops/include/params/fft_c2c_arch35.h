/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef ASDSIP_PARAMS_FFT_C2C_ARCH35_H
#define ASDSIP_PARAMS_FFT_C2C_ARCH35_H

#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {

/**
 * Parameter struct for FftC2C Arch35 (Ascend 950) operation.
 *
 * Fields:
 *   fftN         - Full signal length
 *   batchSize    - Number of batched FFTs
 *   radixListLen - Number of radix decomposition stages (length of plan)
 *   isInverse    - 0 for forward FFT, 1 for inverse FFT
 */
struct FftC2CArch35 {
    int64_t fftN;
    int64_t batchSize;
    int64_t radixListLen;
    int32_t isInverse;

    bool operator==(const FftC2CArch35 &other) const
    {
        return this->fftN == other.fftN && this->batchSize == other.batchSize &&
               this->radixListLen == other.radixListLen && this->isInverse == other.isInverse;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: FftC2CArch35"
           << ", fftN:" << fftN << ", batchSize:" << batchSize
           << ", radixListLen:" << radixListLen << ", isInverse:" << isInverse;
        return ss.str();
    }
};

}  // namespace OpParam
}  // namespace AsdSip

#endif
