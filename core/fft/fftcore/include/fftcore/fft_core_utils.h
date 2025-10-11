/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_CORE_UTILS__
#define __FFT_CORE_UTILS__

#include <cstdint>

constexpr uint64_t DIV_2 = 2;
inline uint64_t GetC2RInputSize(uint64_t fftSize)
{
    return (fftSize / DIV_2) + 1;
}

inline uint64_t GetC2ROutputSize(uint64_t fftSize)
{
    return fftSize;
}

inline uint64_t GetR2CInputSize(uint64_t fftSize)
{
    return fftSize;
}

inline uint64_t GetR2COutputSize(uint64_t fftSize)
{
    return (fftSize / DIV_2) + 1;
}

#endif
