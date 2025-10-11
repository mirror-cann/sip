/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFTCORE_ANY__
#define __FFTCORE_ANY__

#include <mki/utils/rt/rt.h>
#include "ops.h"
#include "fftcore/fft_core_base.h"

class FFTCoreAny : public FFTCoreBase {
public:
    FFTCoreAny(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
               bool forward)
        : FFTCoreBase(FFTCoreType::kAny, nDone, nDoing, nLeft, batch, fftType, forward)
    {
    }
    ~FFTCoreAny() override {}
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor &inTensor, Tensor &outTensor, void *stream, workspace::Workspace &workspace) override;
    void Run(void *input, void *output, void *stream, workspace::Workspace &workspace) override;

private:
    void InitRadix() override;
    bool PreAllocateInDevice() override;
    std::vector<std::shared_ptr<AsdSip::FFTensor>> coeffMatrices;
    uint8_t *buffer = nullptr;
    uint8_t *deviceBuffer = nullptr;
};

#endif