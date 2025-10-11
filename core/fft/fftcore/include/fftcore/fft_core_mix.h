/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_CORE_MIX__
#define __FFT_CORE_MIX__

#include "ops.h"
#include "fftcore/fft_core_mix_base.h"
#include "utils/aspb_status.h"

class FftCoreMix : public FftCoreMixBase {
public:
    FftCoreMix(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
               bool forward)
        : FftCoreMixBase(FFTCoreType::kFftMix, nDone, nDoing, nLeft, batch, fftType, forward, -1, nDoing)
    {
    }

private:
    void InitParams() override;
    std::string GetOpName() override;
    bool PreAllocateInDevice() override;
    void InitLaunchParam() override;
};

#endif