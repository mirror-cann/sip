/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
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

using Mki::SVector;

namespace OpParam {
struct FftB {
    uint32_t nFFT;
    uint32_t batchSize;
    int isForward;
    Mki::SVector<size_t> radixVec;
    bool operator==(const FftB &other) const
    {
        return this->nFFT == other.nFFT && this->batchSize == other.batchSize && this->isForward == other.isForward &&
               this->radixVec == other.radixVec;
    };

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: fft_b"
           << ", nFFT:" << nFFT << ", batchSize:" << batchSize << ", isForward:" << isForward
           << ", radixVec:" << radixVec;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_FFTB_H
