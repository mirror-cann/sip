/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_DFT_C2R_H
#define ASDSIP_PARAMS_DFT_C2R_H

#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {
struct DftC2R {
    int64_t fftN;
    int64_t batchSize;
    int isInverse;

    bool operator==(const DftC2R &other) const
    {
        return this->fftN == other.fftN && this->batchSize == other.batchSize && this->isInverse == other.isInverse;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: DftC2R" << ", batchSize:" << batchSize << ", fftN:" << fftN << ", isInverse:" << isInverse;
        return ss.str();
    }
};
} // namespace OpParam
} // namespace AsdSip

#endif
