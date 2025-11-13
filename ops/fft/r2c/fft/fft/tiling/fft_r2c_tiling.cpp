/**
 * Copyright (c) 2025 Huawei Technologies Co., Ltd.
 * This program is free software, you can redistribute it and/or modify it under the terms and conditions of
 * CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */

#include "fft_r2c_tiling.h"
#include "../../../../common/include/tiling/fft_all_mix_tiling_ops_utils.h"

namespace AsdSip {
using namespace Mki;
AspbStatus FftR2CTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    return InitFftAllMixTiling(launchParam, kernelInfo);
}
} // namespace AsdSip