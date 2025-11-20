/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_SSYR_H
#define ASDSIP_PARAMS_SSYR_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {
namespace OpParam {

constexpr float FLOAT_THRESHOLD = 1e-9;

struct Ssyr {
    uint32_t uplo;
    uint32_t n;
    uint32_t incx;
    float alpha;

    bool operator==(const Ssyr &other) const
    {
        bool isAlphaEqual = abs(this->alpha - other.alpha) < FLOAT_THRESHOLD;
        return isAlphaEqual && this->n == other.n && this->uplo == other.uplo;
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "uplo:" << uplo << ", n:" << n << ", alpha:" << alpha;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip
#endif  // ASDSIP_PARAMS_SSYR_H