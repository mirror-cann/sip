/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#ifndef ASDSIP_PARAMS_CAXPY_H
#define ASDSIP_PARAMS_CAXPY_H

#include <cstdint>
#include <string>
#include <sstream>
#include <mki/utils/SVector/SVector.h>

namespace AsdSip {
namespace OpParam {
struct Caxpy {
    uint32_t n;
    uint32_t incx;
    uint32_t incy;
    float alphaReal;
    float alphaImag;

    bool FloatEqual(float a, float b) const
    {
        float closeTo0 = float(1e-6);
        if (a > b) {
            return a - b < closeTo0;
        } else {
            return b - a < closeTo0;
        }
    };

    bool operator==(const Caxpy &other) const
    {
        float floatN = static_cast<float>(this->n);
        return FloatEqual(floatN, other.alphaReal) && FloatEqual(this->alphaReal, other.alphaImag) &&
               FloatEqual(floatN, other.alphaImag);
    }

    std::string ToString() const
    {
        std::stringstream ss;
        ss << "n: " << n << ", alpha real: " << alphaReal << ", alpha imag: " << alphaImag;
        return ss.str();
    }
};
}  // namespace OpParam
}  // namespace AsdSip

#endif  // ASDSIP_PARAMS_CAXPY_H