/**
 * Copyright (c) 2026 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#ifndef __FFT_C2C_ARCH35_CORE__
#define __FFT_C2C_ARCH35_CORE__

#include <vector>
#include "ops.h"
#include "fftcore/fft_core_base.h"
#include "utils/aspb_status.h"

struct FftC2CPlanStage {
    int64_t radix;
    int64_t M;
};

class FftC2CCoreArch35 : public FFTCoreBase {
public:
    FftC2CCoreArch35(unsigned nDone, unsigned nDoing, unsigned nLeft, unsigned batch, AsdSip::asdFftType fftType,
                     bool forward)
        : FFTCoreBase(FFTCoreType::kFftC2CArch35, nDone, nDoing, nLeft, batch, fftType, forward)
    {
    }
    ~FftC2CCoreArch35() override { DestroyInDevice(); }
    size_t EstimateWorkspaceSize() override;
    void Run(Tensor& input, Tensor& output, void *stream, workspace::Workspace& workspace) override {}
    void Run(void *input, void *output, void *stream, workspace::Workspace &workspace) override;

    static constexpr int64_t ALLOWED_RADICES[] = {2, 3, 5, 7, 11, 13, 17, 19};

private:
    void InitRadix() override;
    bool PreAllocateInDevice() override;
    void DestroyInDevice() const;
    AsdSip::AspbStatus BuildFftPlan();
    AsdSip::AspbStatus InitTactic();

    std::vector<FftC2CPlanStage> plan;
    std::shared_ptr<AsdSip::FFTensor> radixListTensor;
    std::shared_ptr<AsdSip::FFTensor> dftMatrixArray;
    std::shared_ptr<AsdSip::FFTensor> twMatrixArray;

    std::string opName{"FftC2CArch35Operation"};
};

#endif
