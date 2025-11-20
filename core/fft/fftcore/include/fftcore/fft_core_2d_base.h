/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTCORE_2D_BASE__
#define __FFTCORE_2D_BASE__

#include "utils/fftensor_cache.h"
#include "fftoperation/fft_operation.h"

#include "fft_api.h"
#include "fftcore/_fft_core_type.h"

struct Fft2DProblemDesc {
    unsigned fftX;
    unsigned fftY;
    unsigned batchSize;
    AsdSip::asdFftType fft_type;
    bool forward;
};

class FftCore2DBase : public FftOperation {
public:
    FftCore2DBase(FFTCoreType core_type, unsigned fftX, unsigned fftY, unsigned batchSize,
                  AsdSip::asdFftType fft_type, bool forward)
        : problemDesc{fftX, fftY, batchSize, fft_type, forward},
          coreType(core_type)
    {
    }
    bool init() override
    {
        return PreAllocateInDevice();
    }
    ~FftCore2DBase() override {}
    virtual void Run(Tensor &input, Tensor &output, void *stream, workspace::Workspace &workspace) = 0;

protected:
    virtual bool PreAllocateInDevice() = 0;
    const Fft2DProblemDesc problemDesc;
    const FFTCoreType coreType;
};

#endif
