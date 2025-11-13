/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTCORE_BASE__
#define __FFTCORE_BASE__

#include <vector>
#include "utils/fftensor_cache.h"
#include "fftoperation/fft_operation.h"

#include "fft_api.h"
#include "fftcore/_fft_core_type.h"

struct ProblemDesc {
    unsigned nDone;
    unsigned nDoing;
    unsigned nLeft;
    unsigned batch;
    AsdSip::asdFftType fftType;
    bool forward;
};

class FFTCoreBase : public FftOperation {
public:
    FFTCoreBase(FFTCoreType coreType, unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch,
                AsdSip::asdFftType fftType, bool forward)
        : problemDesc{nDone, nDoing, nLeft, batch, fftType, forward},
          coreType(coreType)
    {
    }
    bool init() override
    {
        InitRadix();
        return PreAllocateInDevice();
    }
    ~FFTCoreBase() override {}
    virtual void Run(Tensor &input, Tensor &output, void *stream, workspace::Workspace &workspace) = 0;

protected:
    virtual void InitRadix() {}
    virtual bool PreAllocateInDevice()
    {
        return true;
    }
    const ProblemDesc problemDesc;
    const FFTCoreType coreType;
    std::vector<int64_t> radixVec;
};

#endif
