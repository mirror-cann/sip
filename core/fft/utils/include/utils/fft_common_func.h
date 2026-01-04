/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_COMMON_FUNC_H
#define __FFT_COMMON_FUNC_H

#include <algorithm>

namespace AsdSip {
// private functions

inline int64_t GetC2RInputSize(int64_t fftSize)
{
    return (fftSize / 2) + 1;
}

inline int64_t GetC2ROutputSize(int64_t fftSize)
{
    return fftSize;
}

inline int64_t GetR2CInputSize(int64_t fftSize)
{
    return fftSize;
}

inline int64_t GetR2COutputSize(int64_t fftSize)
{
    return (fftSize / 2) + 1;
}

inline std::vector<int64_t> orderedFactorize(int64_t size)
{
    std::vector<int64_t> factors{};
    if (size <= 0) {
        return factors;
    }

    int64_t bound = static_cast<int64_t>(std::sqrt(size));
    for (int64_t factor = 2; factor <= bound;) {
        if (size % factor == 0) {
            factors.push_back(factor);
            size /= factor;
            bound = static_cast<int64_t>(std::sqrt(size));
        } else {
            factor++;
        }
    }

    if (size != 1) {
        factors.push_back(size);
    }

    return factors;
}
}

#endif