/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_FFTN_H
#define ASDSIP_PARAMS_FFTN_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {

using Mki::SVector;

namespace OpParam {
struct FftN {
    uint32_t nFFT;
    uint32_t batchSize;
    uint32_t repeatBatchSize;
    int isForward;
    Mki::SVector<size_t> radixVec;
    Mki::SVector<size_t> aicInputAddr;
    Mki::SVector<size_t> aivOutputAddr;
    Mki::SVector<size_t> lessTCopy;
    Mki::SVector<size_t> syncTilingNum;
    bool operator==(const FftN &other) const
    {
        return this->nFFT == other.nFFT && this->batchSize == other.batchSize && this->radixVec == other.radixVec;
    };

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: fft_n "
           << ", nFFT:" << nFFT << ", batchSize:" << batchSize << ", isForward:" << isForward
           << ", radixVec:" << radixVec << ", aicInputAddr:" << aicInputAddr << ", aivOutputAddr:" << aivOutputAddr
           << ", lessTCopy:" << lessTCopy << ", syncTilingNum:" << syncTilingNum;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_FFTN_H