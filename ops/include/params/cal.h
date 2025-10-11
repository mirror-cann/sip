/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_CAL_H
#define ASDSIP_PARAMS_CAL_H

#include <cstdint>
#include <string>
#include <sstream>

namespace AsdSip {
namespace OpParam {
struct Cal {
    int64_t n;
    float alphaReal;
    float alphaImag;

    enum class CalType {
        CAL_SSCAL = 0,
        CAL_CSCAL,
        CAL_CSSCAL,
    };

    bool FloatEqual(float a, float b) const
    {
        float closeTo0 = float(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
    };

    CalType calType;
    bool operator==(const Cal &other) const
    {
        return this->n == other.n && FloatEqual(this->alphaReal, other.alphaReal) &&
               FloatEqual(this->alphaImag, other.alphaImag) && this->calType == other.calType;
        ;
    };

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "OpName: Cal"
           << ", n: " << n << ", alphaReal: " << alphaReal << ", alphaImag: " << alphaImag << ", calType:" << static_cast<int>(calType);
        ;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_CAL_H