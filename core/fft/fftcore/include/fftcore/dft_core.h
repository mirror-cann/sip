/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __DFTCORE__
#define __DFTCORE__

#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

using namespace AsdSip;

class DFTCore : public FFTCoreBase {
public:
    DFTCore(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
            bool forward)
        : FFTCoreBase(FFTCoreType::kDft, nDone, nDoing, nLeft, batch, fftType, forward)
    {
    }
    ~DFTCore() override
    {
        DestroyInDevice();
    }
    void Run(Tensor &input, Tensor &output, void *stream, workspace::Workspace &workspace) override;
    size_t EstimateWorkspaceSize() override;

private:
    void InitRadix() override;
    bool PreAllocateInDevice() override;
    void DestroyInDevice() const;
    AspbStatus InitRotationMatrix();
    AspbStatus InitTactic();

    std::string opName{"DftOperation"};
    std::shared_ptr<AsdSip::FFTensor> rotationMatrix;
};

#endif