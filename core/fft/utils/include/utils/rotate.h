/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and contiditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#pragma once

#include "utils/the_tensor.h"
#include "fft_api.h"
#include "utils/fft_mode.h"

namespace rotate {

using FftMode = AsdSip::FftMode;

std::tuple<int64_t, int64_t, int64_t, int64_t> computeOutShape(const std::vector<int64_t> &factors,
                                                               int64_t index, FftMode fftMode);

wten::TheTensor<float> OneRotateMatrix(std::vector<int64_t> factors, int64_t index, FftMode fftMode, bool isForward);

}
