/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_DFT_H
#define ASDSIP_PARAMS_DFT_H

#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>
#include "mki/types.h"
#include "mki/utils/compare/compare.h"

namespace AsdSip {
namespace OpParam {
struct FftStride {
    size_t batchSize;
    size_t fftN;
    size_t strideSize;
    int isInverse;
    Mki::SVector<size_t> radixVec;

    bool operator==(const FftStride &other) const
    {
        return this->fftN == other.fftN && this->batchSize == other.batchSize && this->strideSize == other.strideSize &&
               this->isInverse == other.isInverse;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: FFT_Stride"
           << ", batchSize:" << batchSize << ", fftN:" << fftN << ", stride:" << strideSize
           << ", isInverse:" << isInverse;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif
