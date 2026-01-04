/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_DDD_H
#define ASDSIP_PARAMS_DDD_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {
namespace OpParam {
struct Ddd {
    size_t batchSize;
    size_t fftX;
    size_t fftY;
    size_t fftZ;

    bool operator==(const Ddd &other) const
    {
        return this->batchSize == other.batchSize && this->fftX == other.fftX && this->fftY == other.fftY && this->fftZ == other.fftZ;
    };

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: ddd " << ", batchSize:" << batchSize << ", fftX:" << fftX << ", fftY:" << fftY << ", fftZ:" << fftZ;
        return ss.str();
    }
};
} // namespace OpParam
} // namespace AsdSip

#endif // ASDSIP_PARAMS_DDD_H