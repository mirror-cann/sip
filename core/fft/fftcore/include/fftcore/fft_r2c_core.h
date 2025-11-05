/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_R2C_CORE__
#define __FFT_R2C_CORE__

#include "ops.h"
#include "fftcore/fft_core_mix_base.h"
#include "utils/aspb_status.h"

constexpr unsigned DIV_TWO = 2;
class FftR2CCore : public FftCoreMixBase {
public:
    FftR2CCore(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
               bool forward)
        : FftCoreMixBase(FFTCoreType::kFftR2C, nDone, nDoing, nLeft, batch, fftType, forward,
                         static_cast<bool>(nDoing % DIV_TWO) ? 1 : 0, nDoing % DIV_TWO ? nDoing : nDoing / DIV_TWO)
    {
    }

private:
    std::string GetOpName() override;
    bool PreAllocateInDevice() override;
    void InitLaunchParam() override;
};

#endif