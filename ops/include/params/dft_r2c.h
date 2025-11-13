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

namespace AsdSip {
namespace OpParam {
struct DftR2C {
    size_t fftN;
    size_t batchSize;
    int isInverse;

    bool operator==(const DftR2C &other) const
    {
        return this->fftN == other.fftN && this->batchSize == other.batchSize && this->isInverse == other.isInverse;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: DftR2C"
           << ", batchSize:" << batchSize << ", fftN:" << fftN << ", isInverse:" << isInverse;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif
