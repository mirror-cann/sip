/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_FFTB_H
#define ASDSIP_PARAMS_FFTB_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {
namespace OpParam {
struct FftBC2R {
    uint32_t batchSize;
    uint32_t nFFT;
    int isForward;
    uint32_t C2CFftLength;
    Mki::SVector<size_t> radixVec;
    bool operator==(const FftBC2R &other) const
    {
        return this->nFFT == other.nFFT && this->batchSize == other.batchSize && this->isForward == other.isForward &&
               this->radixVec == other.radixVec;
    };

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: fft_b_c2r"
           << ", nFFT:" << nFFT << ", batchSize:" << batchSize << ", isForward:" << isForward
           << ", radixVec:" << radixVec;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_FFTB_H
