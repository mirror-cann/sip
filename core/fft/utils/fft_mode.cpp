/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "log/log.h"

#include "utils/fft_mode.h"

namespace AsdSip {

FftMode convert(AsdSip::asdFftType fftType)
{
    switch (fftType) {
        case asdFftType::ASCEND_FFT_C2C:
        case asdFftType::ASCEND_STFT_C2C:
            return FftMode::c2c;
        case asdFftType::ASCEND_FFT_C2R:
        case asdFftType::ASCEND_STFT_C2R:
            return FftMode::c2r;
        case asdFftType::ASCEND_FFT_R2C:
        case asdFftType::ASCEND_STFT_R2C:
            return FftMode::r2cOneside;
        default:
            ASDSIP_LOG(ERROR) << "Invalid fft type.";
            throw std::runtime_error("Invalid fft type.");
    }
}

}
