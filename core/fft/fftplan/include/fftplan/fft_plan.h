/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "fftcore/fft_core.h"
#include "fft_api.h"
#include "utils/workspace.h"
namespace AsdSip {

enum class asdFftPlanStatus { PLAN_UNINITIALIZED = 0, PLAN_INITIALIZED = 1, PLAN_FAILED = 2 };

struct PlanStep {
    std::unique_ptr<FftOperation> operation;
};

class FFTPlan {
public:
    asdFftPlanStatus fftStatus = AsdSip::asdFftPlanStatus::PLAN_UNINITIALIZED;
    asdFftType fftType = ASCEND_FFT_C2C;
    asdFftDirection direction = ASCEND_FFT_FORWARD;
    void *stream = nullptr;
    int64_t batchSize = 0;
    std::vector<int64_t> fftSizes;
    std::vector<int64_t> fftStrides;
    std::vector<PlanStep> steps;
    void* workspaceAddr = nullptr;

    FFTPlan();
    void markInitialized();
    void markFailed();
    bool isInitialized() const;
    bool hasWorkspace() const;
    bool isForward() const;
    int64_t eleNum() const;
};

}
