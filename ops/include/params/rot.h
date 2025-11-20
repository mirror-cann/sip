/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_ROT_H
#define ASDSIP_PARAMS_ROT_H

#include <cstdint>
#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {

constexpr float FLOAT_THRESHOLD = 1e-9;

struct Rot {
    int64_t elementCount;
    float cosValue;
    float sinValue;

    enum class RotType {
        ROT_CSROT = 0,
    };

    RotType rotType;
    bool operator==(const Rot &other) const
    {
        bool isCosValueEqual = abs(this->cosValue - other.cosValue) < FLOAT_THRESHOLD;
        bool isSinValueEqual = abs(this->sinValue - other.sinValue) < FLOAT_THRESHOLD;
        return this->elementCount == other.elementCount && isCosValueEqual &&
               isSinValueEqual && this->rotType == other.rotType;
    };

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: Rot"
           << ", elementCount: " << elementCount << ", cosValue: " << cosValue << ", sinValue: " << sinValue
           << ", rotType:" << static_cast<int>(rotType);
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_ROT_H