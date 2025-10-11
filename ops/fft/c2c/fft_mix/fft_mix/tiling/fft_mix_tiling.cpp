/**
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * This file is a part of the CANN Open Software.
 * Licensed under CANN Open Software License Agreement Version 2.0 (the "License").
 * Please refer to the License for details. You may not use this file except in compliance with the License.
 * THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
 * INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
 * See LICENSE in the root of the software repository for the full text of the License.
 */
#include "fft_mix_tiling.h"
#include "../../../../common/include/tiling/fft_all_mix_tiling_ops_utils.h"
#include "utils/aspb_status.h"

namespace AsdSip {
AsdSip::AspbStatus FftMixTiling(const LaunchParam &launchParam, KernelInfo &kernelInfo)
{
    return InitFftAllMixTiling(launchParam, kernelInfo);
}
}  // namespace AsdSip